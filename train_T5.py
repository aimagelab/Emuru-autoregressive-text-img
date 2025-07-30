import argparse
from pathlib import Path
import math
import uuid

from tqdm import tqdm
import torch
import wandb
from PIL import Image
from torchvision.transforms import functional as F
from loguru import logger
from accelerate import Accelerator
from accelerate.utils import broadcast
from accelerate.utils import ProjectConfiguration, set_seed
from transformers.optimization import get_scheduler

from utils import TrainState
from custom_datasets import DataLoaderManager
from models.emuru import Emuru, EmuruConfig


@torch.no_grad()
def validation(eval_loader, model, accelerator, weight_dtype, len_eval_loader, wandb_prefix="eval"):
    model = accelerator.unwrap_model(model)
    model.eval()
    eval_loss = 0.

    for _, batch in enumerate(eval_loader):
        with accelerator.autocast():
            images = batch['img'].to(weight_dtype)
            input_ids = batch['input_ids'].long()

            loss, _, _ = model(images, input_ids=input_ids, attention_mask=batch['attention_mask'])
            eval_loss += loss.item()

    accelerator.log({f"{wandb_prefix}/loss": eval_loss / len_eval_loader,})

    del model
    torch.cuda.empty_cache()
    return eval_loss / len_eval_loader


@torch.no_grad()
def karaoke_test(karaoke_loader, model, accelerator, weight_dtype, wandb_prefix="test"):
    model = accelerator.unwrap_model(model)
    model.eval()
    eval_loss = 0.
    wandb_image_for_log = None

    for i, batch in enumerate(karaoke_loader):
        with accelerator.autocast():
            style_img_text = batch['style_imgs_text'][0]
            gen_text = batch['gen_text'][0]

            res = model.tokenizer(style_img_text, return_tensors='pt', padding=True, return_attention_mask=True, return_length=True)
            style_img = F.to_tensor(batch['style_imgs'][0][0])
            style_img = F.normalize(style_img, [0.5], [0.5]).cuda()[None].to(weight_dtype)

            loss, _, _ = model(style_img, input_ids=res['input_ids'].cuda().long(), attention_mask=res['attention_mask'].cuda())
            eval_loss += loss.item()

            if i == 0:
                generated_pil_image = model.generate(
                    style_text=style_img_text[0],
                    gen_text=gen_text,
                    style_img=style_img,
                    max_new_tokens=64
                )
                style_img_pil = batch['style_imgs'][0][0]
                generated_sample = Image.new('RGB', (style_img_pil.width + generated_pil_image.width, style_img_pil.height), (255, 255, 255))
                generated_sample.paste(style_img_pil, (0, 0))
                generated_sample.paste( Image.new('RGB', (1, style_img_pil.height), (128, 128, 128)), (style_img_pil.width, 0))
                generated_sample.paste(generated_pil_image, (style_img_pil.width, 0))
                wandb_image_for_log = wandb.Image(generated_sample, caption= f"Style: {style_img_text[0]}, Generated: {gen_text}")

    accelerator.log({
        f"{wandb_prefix}/loss": eval_loss / len(karaoke_loader),
        f"{wandb_prefix}/generated_image": wandb_image_for_log
        })

    del model
    torch.cuda.empty_cache()
    return eval_loss / len(karaoke_loader)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results_t5', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results_t5', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=2, help="train batch size") 
    parser.add_argument("--eval_batch_size", type=int, default=8, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of train epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24, help="random seed")
   
    parser.add_argument("--eval_epochs", type=int, default=1, help="eval interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--run_id", type=str, default=uuid.uuid4().hex[:4], help="uuid of the run")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default="emuru-T5", help="wandb project name")
    parser.add_argument('--wandb_log_interval_steps', type=int, default=25, help="wandb log interval")

    parser.add_argument("--vae_path", type=str, default="blowing-up-groundhogs/emuru_vae", help='vae checkpoint path')

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)

    parser.add_argument('--teacher_noise', type=float, default=0.1, help='How much noise add during training')
    parser.add_argument('--training_type', type=str, default='pretrain', help='Pre-training or long lines finetune', choices=['pretrain', 'finetune'])

    args = parser.parse_args()

    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.adam_weight_decay = 0.01

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

    emuru_config = EmuruConfig(
        t5_name_or_path='google-t5/t5-large',
        vae_name_or_path=args.vae_path,
        tokenizer_name_or_path='google/byt5-small',
        slices_per_query=1,
        vae_channels=1
    )
    model = Emuru(emuru_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience}
    )

    if args.training_type == 'pretrain':
        train_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/train/{000000..000498}.tar")
        eval_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/train/{000499..000499}.tar")
        NUM_SAMPLES_TRAIN = 8_000 * 499
        NUM_SAMPLES_EVAL = 8_000
    elif args.training_type == 'finetune':
        train_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/fine_tune/{000000..000048}.tar")
        eval_pattern = ("https://huggingface.co/datasets/blowing-up-groundhogs/font-square-v2/resolve/main/tars/fine_tune/{000049..000049}.tar")
        NUM_SAMPLES_TRAIN = 8_000 * 49
        NUM_SAMPLES_EVAL = 8_000
    else:
        raise ValueError(f"Invalid training type: {args.training_type}")

    data_loader = DataLoaderManager(
        train_pattern=train_pattern,
        eval_pattern=eval_pattern,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=False,
        persistent_workers=False,
        tokenizer=model.tokenizer,
    )
    train_loader = data_loader.create_dataset('train', 't5')
    eval_loader = data_loader.create_dataset('eval', 't5')
    karaoke_loader = data_loader.create_karaoke_dataset()

    LEN_EVAL_LOADER = NUM_SAMPLES_EVAL // args.eval_batch_size

    model, optimizer, train_loader, eval_loader, karaoke_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, eval_loader, karaoke_loader, lr_scheduler)

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

    args.emuru_params = sum([p.numel() for p in model.parameters()])

    logger.info("***** Running T5 training *****")
    logger.info(f"  Num train samples = {NUM_SAMPLES_TRAIN}. Num steps per epoch = {num_steps_per_epoch}")
    logger.info(f"  Num eval samples = {NUM_SAMPLES_EVAL}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total trainable parameters count = {args.emuru_params}")

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

    for epoch in range(train_state.epoch, args.epochs):

        model.train()
        train_loss = 0.

        for batch in train_loader:

            with accelerator.accumulate(model):
                images = batch['img'].to(weight_dtype)
                input_ids = batch['input_ids'].long()

                loss, _, _ = model(images, input_ids=input_ids, attention_mask=batch['attention_mask'], noise=args.teacher_noise)

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

                train_state.global_step += 1
                logs["global_step"] = train_state.global_step
                logs['train/loss'] = train_loss
                train_loss = 0.

            logs["lr"] = optimizer.param_groups[0]['lr']
            logs['epoch'] = epoch

            progress_bar.set_postfix(**logs)
            if train_state.global_step % args.wandb_log_interval_steps == 0:
                accelerator.log(logs)

        train_state.epoch += 1

        if epoch % args.eval_epochs == 0 and accelerator.is_main_process:
            with torch.no_grad():
                eval_loss = validation(eval_loader, model, accelerator, weight_dtype, LEN_EVAL_LOADER, 'eval')
                eval_loss = broadcast(torch.tensor(eval_loss, device=accelerator.device), from_process=0)

                if eval_loss < train_state.best_eval:
                    train_state.best_eval = eval_loss
                    model_to_save = accelerator.unwrap_model(model)
                    model_to_save.save_pretrained(args.output_dir / f"model_{epoch:04d}")
                    del model_to_save
                    logger.info(f"Epoch {epoch} - Best eval loss: {eval_loss}")
                
                train_state.last_eval = eval_loss

                test_loss = karaoke_test(karaoke_loader, model, accelerator, weight_dtype, 'test')
                logger.info(f"Epoch {epoch} - Test loss: {test_loss}")
                accelerator.save_state()

            lr_scheduler.step(eval_loss)
            accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(args.output_dir)

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()
