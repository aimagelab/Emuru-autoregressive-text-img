from typing import Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel


class WriterIDConfig(PretrainedConfig):
    model_type = "writer_id"
    
    def __init__(
        self,
        num_writers: int = 115961,
        in_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (64, 128, 256),
        latent_channels: int = 256,
        norm_num_groups: int = 16,
        encoder_dropout: float = 0.1,
        **kwargs
    ):
        self.num_writers = num_writers
        self.in_channels = in_channels
        self.block_out_channels = block_out_channels
        self.latent_channels = latent_channels
        self.norm_num_groups = norm_num_groups
        self.encoder_dropout = encoder_dropout
        super().__init__(**kwargs)


class WriterID(PreTrainedModel):
    """
    WriterID model for writer identification from text images.
    """
    
    config_class = WriterIDConfig  # type: ignore

    def __init__(self, config):
        super().__init__(config)

        self.conv_in = nn.Conv2d(
            config.in_channels,
            config.block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])
        output_channel = config.block_out_channels[0]
        num_blocks = len(config.block_out_channels) - 1

        for i in range(1, num_blocks):
            input_channel = output_channel
            output_channel = config.block_out_channels[i]
            is_final_block = i == num_blocks - 1
            is_second_to_last_block = i == num_blocks - 2

            self.down_blocks.append(
                ResnetBlock(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=config.encoder_dropout,
                    group_norm=config.norm_num_groups))

            if is_final_block:
                self.down_blocks.append(Downsample(output_channel, True, down_sample_factor=(1, 2)))
            elif is_second_to_last_block:
                self.down_blocks.append(Downsample(output_channel, True, down_sample_factor=(2, 4)))
            else:
                self.down_blocks.append(Downsample(output_channel, True, down_sample_factor=(4, 4)))

        self.down_blocks.append(
            ResnetBlock(
                in_channels=config.block_out_channels[-1],
                out_channels=config.latent_channels,
                temb_channels=0,
                dropout=config.encoder_dropout,
                group_norm=config.norm_num_groups))

        self.down_blocks = nn.Sequential(*self.down_blocks)

        self.linear = nn.Linear(config.latent_channels, config.latent_channels)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(config.latent_channels, config.num_writers)


    def forward(self, x):
        """Forward pass through the network."""
        x = self.compute_features(x)
        out = torch.mean(x, dim=[-1, -2])
        out = self.linear(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    

    def compute_features(self, x):
        """Compute feature representation."""
        x = self.conv_in(x)
        x = self.down_blocks(x)
        return x


    def remove_last_layers(self):
        """Remove classification layers for feature extraction."""
        if hasattr(self, 'linear'):
            del self.linear
        if hasattr(self, 'relu'):
            del self.relu
        if hasattr(self, 'linear2'):
            del self.linear2
    
    
    def reset_last_layer(self, num_writers: int):
        """Reset the final classification layer for a different number of writers."""
        latent_channels = self.config.get('latent_channels', 256)
        self.linear2 = nn.Linear(latent_channels, num_writers)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, group_norm=16):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = get_group_norm(num_groups=group_norm, num_channels=in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = get_group_norm(num_groups=group_norm, num_channels=out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1)

        self.non_linearity = nn.SiLU()
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool, down_sample_factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=down_sample_factor,
                padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def get_group_norm(num_channels, num_groups=32):
    if num_channels < num_groups:
        num_groups = num_channels
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-5, affine=True)
