from typing import  Tuple

import torch.nn as nn
from einops import rearrange
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .vae import Encoder
from .nn_utils import PositionalEncoding1D, A2DPE


class HTRConfig(PretrainedConfig):
    model_type = "htr"

    def __init__(self,
                 alphabet_size: int = 169,
                 in_channels: int = 3,
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 block_out_channels: Tuple[int] = (64,),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 128,
                 d_model: int = 128,
                 norm_num_groups: int = 16,
                 encoder_dropout: float = 0.1,
                 use_tgt_pe=True,
                 use_mem_pe=True,
                 htr_dropout: float = 0.1,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 4,
                 only_head: bool = False,
                 **kwargs):
        self.alphabet_size = alphabet_size
        self.in_channels = in_channels
        self.down_block_types = down_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.act_fn = act_fn
        self.latent_channels = latent_channels
        self.d_model = d_model
        self.norm_num_groups = norm_num_groups
        self.encoder_dropout = encoder_dropout
        self.use_tgt_pe = use_tgt_pe
        self.use_mem_pe = use_mem_pe
        self.htr_dropout = htr_dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.only_head = only_head
        super().__init__(**kwargs)


class HTR(PreTrainedModel):

    config_class = HTRConfig  # type: ignore

    def __init__(self, config):
        super(HTR, self).__init__(config)

        self.config = config

        if not self.config.only_head:
            self.feature_extractor = Encoder(
                in_channels=self.config.in_channels,
                out_channels=self.config.latent_channels,
                down_block_types=self.config.down_block_types,
                block_out_channels=self.config.block_out_channels,
                layers_per_block=self.config.layers_per_block,
                act_fn=self.config.act_fn,
                norm_num_groups=self.config.norm_num_groups,
                double_z=False,
                dropout=self.config.encoder_dropout,
            )

        self.quant_conv = nn.Conv2d(self.config.latent_channels, self.config.d_model, 1)
        self.text_embedding = nn.Embedding(self.config.alphabet_size, self.config.d_model)
        self.d_model = self.config.d_model
        self.mem_pe = A2DPE(d_model=self.config.d_model, dropout=self.config.htr_dropout) if self.config.use_mem_pe else None
        self.tgt_pe = PositionalEncoding1D(d_model=self.config.d_model, dropout=self.config.htr_dropout) if self.config.use_tgt_pe else None

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.d_model, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.config.num_encoder_layers, norm=nn.LayerNorm(self.config.d_model), enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.d_model, nhead=1)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.config.num_decoder_layers, norm=nn.LayerNorm(self.config.d_model))

        self.fc = nn.Linear(self.config.d_model, self.config.alphabet_size)


    def forward(self, x, tgt_logits, tgt_mask, tgt_key_padding_mask):
        memory = self.compute_features(x)
        tgt = self.text_embedding(tgt_logits)
        if self.config.use_tgt_pe:
            tgt = self.tgt_pe(tgt)

        tgt = rearrange(tgt, "b s d -> s b d")
        if tgt_mask is not None and tgt_mask.dim() == 2 and tgt_mask.size(0) != tgt_mask.size(1):
            tgt_mask = None
        tgt = self.transformer_decoder(tgt, memory, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        tgt = rearrange(tgt, "s b d -> b s d")
        tgt = self.fc(tgt)

        return tgt
    
    
    def compute_features(self, x):
        if not self.config.only_head:
            memory = self.feature_extractor(x)  
        else:
            memory = x

        memory = self.quant_conv(memory)
        if self.config.use_mem_pe:
            memory = self.mem_pe(memory)
        memory = rearrange(memory, "b c h w -> (h w) b c")
        memory = self.transformer_encoder(memory)
            
        return memory
    

    def remove_last_layers(self):
        del self.transformer_decoder
        del self.fc


    def reset_last_layer(self, alphabet_size):
        self.fc = nn.Linear(self.config.d_model, alphabet_size)
