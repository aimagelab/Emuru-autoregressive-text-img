import argparse
from pathlib import Path
import math
import uuid

import wandb
from tqdm import tqdm
import torch
from torchvision.utils import make_grid
from loguru import logger
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from accelerate.utils import broadcast
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.optimization import get_scheduler

from utils import TrainState
from models.htr import HTR
from models.writer_id import WriterID
from custom_datasets import DataLoaderManager
from models.autoencoder_loss import AutoencoderLoss 
from models.autoencoder_kl import AutoencoderKL


@torch.no_grad()
def validation(eval_loader, vae, accelerator, loss_fn, weight_dtype, htr, writer_id, len_eval_loader, wandb_prefix="eval"):
    vae_model = accelerator.unwrap_model(vae)
    vae_model.eval()
    htr_model = accelerator.unwrap_model(htr)
    htr_model.eval()
    writer_id_model = accelerator.unwrap_model(writer_id)
    writer_id_model.eval()
    eval_loss = 0.
    images_for_log = []
    images_for_log_w_htr_wid = []

    for step, batch in enumerate(eval_loader):
        with accelerator.autocast():
            images = batch['rgb'].to(weight_dtype)
            target_images = batch['bw'].to(weight_dtype)
            authors_id = batch['writer_id']

            text_logits_s2s = batch['text_logits_s2s']
            text_logits_s2s_unpadded_len = batch['texts_len']
            tgt_mask = batch['tgt_key_mask']
            tgt_key_padding_mask = batch['tgt_key_padding_mask']

            posterior = vae_model.encode(images).latent_dist
            z = posterior.sample()
            pred = vae_model.decode(z).sample

            loss, log_dict, wandb_media_log = loss_fn(images=target_images, z=z, reconstructions=pred, posteriors=posterior,
                                                      writers=authors_id, text_logits_s2s=text_logits_s2s,
                                                      text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                      tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                      split=wandb_prefix, htr=htr_model, writer_id=writer_id_model)

            eval_loss += loss['loss'].item()

            if step < 2:
                images = ((images + 1) / 2).clamp(0, 1) * 255
                pred = ((pred + 1) / 2).repeat(1, 3, 1, 1).clamp(0, 1) * 255

                author_id = batch['writer_id'][0].item()
                pred_author_id = wandb_media_log[f'{wandb_prefix}/predicted_authors'][0][0]
                text = loss_fn.alphabet.decode(batch['text_logits_s2s'][:, 1:], [loss_fn.alphabet.eos])[0]
                pred_text = wandb_media_log[f'{wandb_prefix}/predicted_characters'][0][0]
                images_for_log_w_htr_wid.append(wandb.Image(
                    torch.cat([images[0], pred[0]], dim=-1),
                    caption=f'AID: {author_id}, Pred AID: {pred_author_id}, Text: {text}, Pred Text: {pred_text}')
                )   

            if step == 0:
                images_for_log.append(torch.cat([images, pred], dim=-1)[:8])         

    accelerator.log({
        **log_dict,
        f"{wandb_prefix}/loss": eval_loss / len_eval_loader,
        "Original (left), Reconstruction (right)": [wandb.Image(make_grid(image, nrow=2, normalize=True, value_range=(0, 255))) for _, image in enumerate(images_for_log)],
        "Image, HTR and Writer ID": images_for_log_w_htr_wid
    })

    del vae_model
    del images_for_log, images_for_log_w_htr_wid
    torch.cuda.empty_cache()
    return eval_loss / len_eval_loader


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results_vae', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results_vae', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of train epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
   
    parser.add_argument("--eval_epochs", type=int, default=1, help="eval interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--run_id", type=str, default=uuid.uuid4().hex[:4], help="uuid of the run")
    parser.add_argument("--vae_config", type=str, default='configs/vae/VAE_64x768.json', help='vae config path')
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default="emuru_vae", help="wandb project name")
    parser.add_argument('--wandb_log_interval_steps', type=int, default=25, help="wandb log interval")

    parser.add_argument("--htr_path", type=str, default='pretrained_models/emuru_vae_htr', help='htr checkpoint path')
    parser.add_argument("--writer_id_path", type=str, default='pretrained_models/emuru_vae_writer_id', help='writerid checkpoint path')

    parser.add_argument("--lr_scheduler", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--use_ema", type=str, default="False")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)

    args = parser.parse_args()

    args.use_ema = args.use_ema == "True"
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.adam_weight_decay = 0

    args.run_name = args.resume_id if args.resume_id else args.run_id
    args.output_dir = Path(args.output_dir) / args.run_name
    args.logging_dir = Path(args.logging_dir) / args.run_name

    accelerator_project_config = ProjectConfiguration(
        project_dir=str(args.output_dir),
        logging_dir=str(args.logging_dir),
        automatic_checkpoint_naming=True,
        total_limit=args.checkpoints_total_limit,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        cpu=False,
    )

    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.logging_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_config(args.vae_config)
    vae.train()
    vae.requires_grad_(True)
    args.vae_params = vae.num_parameters(only_trainable=True)

    if args.use_ema:
        ema_vae = AutoencoderKL.from_config(args.vae_config)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=vae.config)
        accelerator.register_for_checkpointing(ema_vae)

    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    data_loader = DataLoaderManager(
        train_pattern=("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/train/{000000..000498}.tar"),
        eval_pattern=("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/train/{000499..000499}.tar"),
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
    )
    train_loader = data_loader.create_dataset('train', 'vae')
    eval_loader = data_loader.create_dataset('eval', 'vae')

    try: 
        NUM_SAMPLES_TRAIN = len(train_loader.dataset)
        NUM_SAMPLES_EVAL = len(eval_loader.dataset)
    except TypeError:
        NUM_SAMPLES_TRAIN = 8_000 * 499
        NUM_SAMPLES_EVAL = 8_000

    LEN_EVAL_LOADER = NUM_SAMPLES_EVAL // args.eval_batch_size

    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, scheduler_specific_kwargs={"patience": args.lr_scheduler_patience, 'mode': 'min'})

    htr = HTR.from_pretrained(args.htr_path)
    writer_id = WriterID.from_pretrained(args.writer_id_path)
    htr.eval()
    writer_id.eval()
    for param in htr.parameters():
        param.requires_grad = False
    for param in writer_id.parameters():
        param.requires_grad = False

    loss_fn = AutoencoderLoss(alphabet=data_loader.alphabet)
    args.htr_params = sum([p.numel() for p in htr.parameters()])
    args.writer_id_params = sum([p.numel() for p in writer_id.parameters()])
    args.total_params = args.vae_params + args.htr_params + args.writer_id_params

    vae, htr, writer_id, optimizer, train_loader, eval_loader, lr_scheduler, loss_fn = accelerator.prepare(
            vae, htr, writer_id, optimizer, train_loader, eval_loader, lr_scheduler, loss_fn)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        wandb_args = {"wandb": {"entity": args.wandb_entity, "name": args.run_name}}
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.wandb_project_name, tracker_config, wandb_args)

    num_steps_per_epoch = math.ceil(NUM_SAMPLES_TRAIN / (args.train_batch_size * args.gradient_accumulation_steps))
    args.max_train_steps = args.epochs * num_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** Running VAE training *****")
    logger.info(f"  Num train samples = {NUM_SAMPLES_TRAIN}. Num steps per epoch = {num_steps_per_epoch}")
    logger.info(f"  Num eval samples = {NUM_SAMPLES_EVAL}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total trainable parameters count = {args.vae_params}. VAE: {args.vae_params}, HTR: {args.htr_params}, WriterID: {args.writer_id_params}")

    train_state = TrainState(global_step=0, epoch=0, best_eval_init=float('inf'))
    accelerator.register_for_checkpointing(train_state)
    if args.resume_id:
        try:
            accelerator.load_state()
            accelerator.project_configuration.iteration = train_state.epoch
            logger.info(f"  Resuming from checkpoint at epoch {train_state.epoch}")
        except FileNotFoundError as e:
            logger.warning(f"  Checkpoint not found: {e}. Creating a new run")

    progress_bar = tqdm(range(train_state.global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Determine if we need to use .module for multi-process training
    use_module = accelerator.num_processes > 1

    for epoch in range(train_state.epoch, args.epochs):

        vae.train()
        train_loss = 0.

        for batch in train_loader:

            with accelerator.accumulate(vae):
                images = batch['rgb'].to(weight_dtype)
                target_images = batch['bw'].to(weight_dtype)
                authors_id = batch['writer_id']

                text_logits_s2s = batch['text_logits_s2s']
                text_logits_s2s_unpadded_len = batch['texts_len']
                tgt_mask = batch['tgt_key_mask']
                tgt_key_padding_mask = batch['tgt_key_padding_mask']

                if use_module:
                    posterior = vae.module.encode(images).latent_dist
                else:
                    posterior = vae.encode(images).latent_dist
                z = posterior.sample()
                if use_module:
                    pred = vae.module.decode(z).sample
                else:
                    pred = vae.decode(z).sample

                loss, _, _ = loss_fn(images=target_images, z=z, reconstructions=pred, posteriors=posterior,
                                                writers=authors_id, text_logits_s2s=text_logits_s2s,
                                                text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                split="train", htr=htr, writer_id=writer_id)

                loss = loss['loss']

                if not torch.isfinite(loss):
                    logger.warning("non-finite loss")
                    optimizer.zero_grad()
                    continue

                avg_loss = accelerator.gather(loss).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            logs = {}
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.use_ema:
                    ema_vae.to(vae.device)
                    ema_vae.step(vae.parameters())

                train_state.global_step += 1
                logs['train/loss'] = train_loss
                logs["global_step"] = train_state.global_step
                train_loss = 0.

            logs["lr"] = optimizer.param_groups[0]['lr']
            logs['epoch'] = epoch

            progress_bar.set_postfix(**logs)
            if train_state.global_step % args.wandb_log_interval_steps == 0:
                accelerator.log(logs)

        train_state.epoch += 1

        if epoch % args.eval_epochs == 0 and accelerator.is_main_process:
            with torch.no_grad():
                eval_loss = validation(eval_loader, vae, accelerator, loss_fn, weight_dtype,  htr, writer_id, LEN_EVAL_LOADER, 'eval')
                eval_loss = broadcast(torch.tensor(eval_loss, device=accelerator.device), from_process=0)

                if args.use_ema:
                    ema_vae.store(vae.parameters())
                    ema_vae.copy_to(vae.parameters())
                    _ = validation(eval_loader, vae, accelerator, loss_fn, weight_dtype,  htr, writer_id, LEN_EVAL_LOADER, 'ema')
                    ema_vae.restore(vae.parameters())

                if eval_loss < train_state.best_eval:
                    train_state.best_eval = eval_loss
                    vae_to_save = accelerator.unwrap_model(vae)
                    vae_to_save.save_pretrained(args.output_dir / f"model_{epoch:04d}")
                    del vae_to_save
                    logger.info(f"Epoch {epoch} - Best eval loss: {eval_loss}")
                
                train_state.last_eval = eval_loss
                accelerator.save_state()

            accelerator.wait_for_everyone()
            lr_scheduler.step(train_state.last_eval)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
            vae.save_pretrained(args.output_dir / f"ema")

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()
