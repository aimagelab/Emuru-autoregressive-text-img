import torch
import torch.nn as nn
from transformers import PreTrainedModel, T5ForConditionalGeneration, T5Config, AutoTokenizer
from diffusers import AutoencoderKL
from einops.layers.torch import Rearrange
from einops import repeat
from torchvision.transforms import functional as F
from typing import Optional, Tuple, List, Any
from PIL import Image
from transformers import PretrainedConfig
from torchvision.utils import make_grid

class EmuruConfig(PretrainedConfig):
    model_type = "emuru"

    def __init__(self, 
                 t5_name_or_path='google-t5/t5-large', 
                 vae_name_or_path='blowing-up-groundhogs/emuru_vae',
                 tokenizer_name_or_path='google/byt5-small',
                 slices_per_query=1,
                 vae_channels=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.t5_name_or_path = t5_name_or_path
        self.vae_name_or_path = vae_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.slices_per_query = slices_per_query
        self.vae_channels = vae_channels

class Emuru(PreTrainedModel):
    """
    Emuru is a conditional generative model that integrates a T5-based decoder with a VAE
    for image generation conditioned on text and style images.
    Attributes:
        config_class (Type): Configuration class for the model.
        tokenizer (AutoTokenizer): Tokenizer loaded from the provided tokenizer configuration.
        T5 (T5ForConditionalGeneration): T5 model adapted for conditional generation.
        sos (nn.Embedding): Start-of-sequence embedding.
        vae_to_t5 (nn.Linear): Linear projection from VAE latent space to T5 hidden space.
        t5_to_vae (nn.Linear): Linear projection from T5 hidden space back to VAE latent space.
        padding_token (nn.Parameter): Non-trainable parameter for padding tokens.
        padding_token_threshold (nn.Parameter): Non-trainable parameter for padding token threshold.
        vae (AutoencoderKL): Pre-trained Variational Autoencoder.
        query_rearrange (Rearrange): Layer to rearrange VAE latent representations for queries.
        z_rearrange (Rearrange): Layer to rearrange T5 outputs back to VAE latent dimensions.
        mse_criterion (nn.MSELoss): Mean squared error loss function.
    """
    config_class = EmuruConfig

    def __init__(self, config: EmuruConfig) -> None:
        """
        Initialize the Emuru model.
        Args:
            config (EmuruConfig): Configuration object containing model hyperparameters and paths.
        """
        super().__init__(config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)

        t5_config = T5Config.from_pretrained(config.t5_name_or_path)
        t5_config.vocab_size = len(self.tokenizer)
        self.T5 = T5ForConditionalGeneration(t5_config)
        self.T5.lm_head = nn.Identity()
        self.sos = nn.Embedding(1, t5_config.d_model)

        vae_latent_size = 8 * config.vae_channels * config.slices_per_query
        self.vae_to_t5 = nn.Linear(vae_latent_size, t5_config.d_model)
        self.t5_to_vae = nn.Linear(t5_config.d_model, vae_latent_size, bias=False)

        self.padding_token = nn.Parameter( torch.tensor([[-0.4951,  0.8021,  0.3429,  0.5622,  0.5271,  0.5756,  0.7194,  0.6150]]), requires_grad=False)
        self.padding_token_threshold = nn.Parameter(torch.tensor(0.484982096850872), requires_grad=False)

        self.query_rearrange = Rearrange('b c h (w q) -> b w (q c h)', q=config.slices_per_query)
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=config.vae_channels, q=config.slices_per_query)

        self.mse_criterion = nn.MSELoss()
        self.init_weights()

        self.vae = AutoencoderKL.from_pretrained(config.vae_name_or_path)
        self.set_training(self.vae, False)

    def set_training(self, model: nn.Module, training: bool) -> None:
        """
        Set the training mode for a given model and freeze/unfreeze parameters accordingly.
        Args:
            model (nn.Module): The model to set the training mode for.
            training (bool): If True, set the model to training mode; otherwise, evaluation mode.
        """
        model.train() if training else model.eval()
        for param in model.parameters():
            param.requires_grad = training

    def forward(
        self,
        img: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        noise: float = 0,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            img (Optional[torch.Tensor]): Input image tensor.
            input_ids (Optional[torch.Tensor]): Tokenized input IDs.
            attention_mask (Optional[torch.Tensor]): Attention mask for the inputs.
            noise (float): Amount of noise to add in image encoding.
            **kwargs: Additional arguments.
        Returns:
            Tuple containing:
                - mse_loss (torch.Tensor): Mean squared error loss.
                - pred_latent (torch.Tensor): Predicted latent representations.
                - z (torch.Tensor): Sampled latent vector from VAE.
        """
        decoder_inputs_embeds, z_sequence, z = self._img_encode(img, noise)

        output = self.T5(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds)
        vae_latent = self.t5_to_vae(output.logits[:, :-1])
        pred_latent = self.z_rearrange(vae_latent)

        # Fix: Ensure sequence lengths match for loss computation
        min_seq_len = min(vae_latent.size(1), z_sequence.size(1))
        vae_latent_trimmed = vae_latent[:, :min_seq_len]
        z_sequence_trimmed = z_sequence[:, :min_seq_len]
        
        mse_loss = self.mse_criterion(vae_latent_trimmed, z_sequence_trimmed)
        return mse_loss, pred_latent, z

    @torch.inference_mode()
    def generate(
        self,
        style_text: str,
        gen_text: str,
        style_img: torch.Tensor,
        **kwargs: Any
    ) -> Image.Image:
        """
        Generate an image by combining style and generation texts with a style image.
        Args:
            style_text (str): Style-related text prompt.
            gen_text (str): Generation-related text prompt.
            style_img (torch.Tensor): Style image tensor. Expected shape is either 3D or 4D.
            **kwargs: Additional keyword arguments.
        Returns:
            Image.Image: Generated image as a PIL image.
        """
        if style_img.ndim == 3:
            style_img = style_img.unsqueeze(0)
        elif style_img.ndim == 4:
            pass
        else:
            raise ValueError('style_img must be 3D or 4D')
        
        texts = [style_text + ' ' + gen_text]
        imgs, _, img_ends = self._generate(texts=texts, imgs=style_img, **kwargs)
        imgs = (imgs + 1) / 2
        return F.to_pil_image(imgs[0, ..., style_img.size(-1):img_ends.item()].detach().cpu())
    
    @torch.inference_mode()
    def generate_batch(
        self,
        style_texts: List[str],
        gen_texts: List[str],
        style_imgs: torch.Tensor,
        lengths: List[int],
        **kwargs: Any
    ) -> List[Image.Image]:
        """
        Generate a batch of images from lists of style texts, generation texts, and style images.
        Args:
            style_texts (List[str]): List of style-related text prompts.
            gen_texts (List[str]): List of generation-related text prompts.
            style_imgs (torch.Tensor): Batch of style images (4D tensor).
            lengths (List[int]): List of lengths corresponding to each image.
            **kwargs: Additional keyword arguments.
        Returns:
            List[Image.Image]: List of generated images as PIL images.
        """
        assert style_imgs.ndim == 4, 'style_imgs must be 4D'
        assert len(style_texts) == len(style_imgs), 'style_texts and style_imgs must have the same length'
        assert len(gen_texts) == len(style_imgs), 'gen_texts and style_imgs must have the same length'
        texts = [style_text + ' ' + gen_text for style_text, gen_text in zip(style_texts, gen_texts)]
        
        imgs, _, img_ends = self._generate(texts=texts, imgs=style_imgs, lengths=lengths, **kwargs)
        imgs = (imgs + 1) / 2

        out_imgs = []
        for i, end in enumerate(img_ends):
            start = lengths[i]
            out_imgs.append(F.to_pil_image(imgs[i, ..., start:end].detach().cpu()))
        return out_imgs

    def _generate(
        self,
        texts: Optional[List[str]] = None,
        imgs: Optional[torch.Tensor] = None,
        lengths: Optional[List[int]] = None,
        input_ids: Optional[torch.Tensor] = None,
        z_sequence: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        stopping_criteria: str = 'latent',
        stopping_after: int = 10,
        stopping_patience: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal generation routine that combines textual and visual inputs to iteratively generate
        latent representations and decode them into images.
        Args:
            texts (Optional[List[str]]): List of text prompts.
            imgs (Optional[torch.Tensor]): Input image tensor.
            lengths (Optional[List[int]]): Desired lengths for each image in latent space.
            input_ids (Optional[torch.Tensor]): Tokenized input IDs.
            z_sequence (Optional[torch.Tensor]): Precomputed latent sequence.
            max_new_tokens (int): Maximum tokens to generate.
            stopping_criteria (str): Criteria for stopping ('latent' or 'none').
            stopping_after (int): Number of tokens to check for stopping condition.
            stopping_patience (int): Patience parameter for stopping condition.
        Returns:
            Tuple containing:
                - imgs (torch.Tensor): Generated images.
                - canvas_sequence (torch.Tensor): Generated latent canvas sequence.
                - img_ends (torch.Tensor): End indices for each generated image.
        """
        assert texts is not None or input_ids is not None, 'Either texts or input_ids must be provided'
        assert imgs is not None or z_sequence is not None, 'Either imgs or z_sequence must be provided'

        if input_ids is None:
            input_ids = self.tokenizer(texts, return_tensors='pt', padding=True).input_ids
            input_ids = input_ids.to(self.device)

        if z_sequence is None:
            _, z_sequence, _ = self._img_encode(imgs)
        
        if lengths is None:
            lengths = [imgs.size(-1)] * imgs.size(0)
        lengths = torch.tensor(lengths).to(self.device)
        lengths = (lengths / 8).ceil().int()

        z_sequence_mask = torch.zeros((z_sequence.size(0), lengths.max() + max_new_tokens))
        z_sequence_mask = z_sequence_mask.bool().to(self.device)
        for i, l in enumerate(lengths):
            z_sequence_mask[i, :l] = True

        canvas_sequence = z_sequence[:, :lengths.min()]
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=input_ids.size(0))
        pad_token = repeat(self.padding_token, '1 d -> b 1 d', b=input_ids.size(0))
        seq_stops = torch.ones(z_sequence.size(0), dtype=torch.int) * -1

        for token_idx in range(lengths.min(), lengths.max() + max_new_tokens):
            if len(z_sequence) == 0:
                decoder_inputs_embeds = sos
            else:
                decoder_inputs_embeds = self.vae_to_t5(canvas_sequence)
                decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
            output = self.T5(input_ids, decoder_inputs_embeds=decoder_inputs_embeds)
            vae_latent = self.t5_to_vae(output.logits[:, -1:])

            mask_slice = z_sequence_mask[:, token_idx].unsqueeze(-1)
            if token_idx < z_sequence.size(1):
                seq_slice = torch.where(mask_slice, z_sequence[:, token_idx], vae_latent[:, 0])
            else:
                seq_slice = vae_latent[:, 0]
            canvas_sequence = torch.cat([canvas_sequence, seq_slice.unsqueeze(1)], dim=1)

            if stopping_criteria == 'latent':
                similarity = torch.nn.functional.cosine_similarity(canvas_sequence, pad_token, dim=-1)
                windows = (similarity > self.padding_token_threshold).unfold(1, min(stopping_after, similarity.size(-1)), 1)
                window_sums = windows.to(torch.int).sum(dim=2)

                for i in range(similarity.size(0)):
                    idx = (window_sums[i] > (stopping_after - stopping_patience)).nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        seq_stops[i] = idx[0].item()

                if torch.all(seq_stops >= 0):
                    break
            elif stopping_criteria == 'none':
                pass

        imgs = torch.clamp(self.vae.decode(self.z_rearrange(canvas_sequence)).sample, -1, 1)
        return imgs, canvas_sequence, seq_stops * 8
    
    def _img_encode(
        self,
        img: torch.Tensor,
        noise: float = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode the input image into a latent representation using the VAE.
        Args:
            img (torch.Tensor): Input image tensor.
            noise (float): Standard deviation of noise to add to the latent sequence.
        Returns:
            Tuple containing:
                - decoder_inputs_embeds (torch.Tensor): Embeddings to be used as T5 decoder inputs.
                - z_sequence (torch.Tensor): Rearranged latent sequence from the VAE.
                - z (torch.Tensor): Sampled latent vector from the VAE.
        """
        posterior = self.vae.encode(img.float())
        z = posterior.latent_dist.sample()
        z_sequence = self.query_rearrange(z)

        noise_sequence = z_sequence
        if noise > 0:
            noise_sequence = z_sequence + torch.randn_like(z_sequence) * noise

        decoder_inputs_embeds = self.vae_to_t5(noise_sequence)
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=decoder_inputs_embeds.size(0))
        decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
        return decoder_inputs_embeds, z_sequence, z

    def compute_padding_token(self) -> None:
        """
        Compute and update the padding token.
        Raises:
            NotImplementedError: This method must be implemented.
        """
        raise NotImplementedError("compute_padding_token not implemented")

    def compute_padding_token_threshold(self) -> None:
        """
        Compute and update the padding token threshold.
        Raises:
            NotImplementedError: This method must be implemented.
        """
        raise NotImplementedError("compute_padding_token_threshold not implemented")
