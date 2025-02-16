import torch
from torch import nn
from typing import Any
from .utils import PositionalEncoder
from .base import DDPMBase


class ChannelWiseConv2d(nn.Conv2d):
    """A module of a `1 x 1` convolutional network"""

    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__(*(in_channels, out_channels, 1), **kwargs)  # type: ignore


class CBRLayer(nn.Module):
    """A Layer which consists of Conv, BatchNormalizaiton and Relu."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        self._net = nn.Sequential(conv, bn, relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x)
        return out


class ConvBlock(nn.Module):
    """A block which consists of `CBRLayer`"""

    def __init__(
        self, in_channels: int, out_channels: int, num_layers: int = 2
    ) -> None:
        super().__init__()

        net_list: list[nn.Module] = []
        net_list.append(CBRLayer(in_channels=in_channels, out_channels=out_channels))
        for _ in range(num_layers - 1):
            net_list.append(
                CBRLayer(in_channels=out_channels, out_channels=out_channels)
            )
        self._net = nn.Sequential(*net_list)
        self._in_channels = in_channels
        self._out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x)
        return out

    @property
    def in_channels(self):
        return self._in_channels

    @property
    def out_channels(self):
        return self._out_channels


class PosEncConvBlock(nn.Module):
    def __init__(
        self,
        block: ConvBlock,
        timestep_encoded_dim: int,
    ) -> None:
        """A Block which adds the timestep encoding feature to a given block

        Parameters
        ----------
        block : ConvBlock
            A target block
        timestep_encoded_dim : int
            The number of dimensions of timestep encoding.
        """
        super().__init__()
        self._timestep_mlp = nn.Sequential(
            nn.Linear(timestep_encoded_dim, block.in_channels),
            nn.ReLU(),
            nn.Linear(block.in_channels, block.in_channels),
        )
        self._block = block

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_feat: torch.Tensor = self._timestep_mlp(timesteps)
        N, C, _, _ = x.shape
        timestep_feat = timestep_feat.view(N, C, 1, 1)
        feat = x + timestep_feat
        out = self._block(feat)
        return out


class UNet(nn.Module):
    """An example of simple UNet.

    NOTE: This model is not used for training DDPM,
    but is provided for reference purposes only.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()

        self._down_block1 = ConvBlock(num_channels, 64)
        self._down_block2 = ConvBlock(64, 128)
        self._mid_block = ConvBlock(128, 256)
        self._up_block2 = ConvBlock(256 + 128, 128)
        self._up_block1 = ConvBlock(128 + 64, 64)
        self._out_block = ChannelWiseConv2d(64, num_channels)

        self._maxpool = nn.MaxPool2d(2)
        self._upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        res_feat1 = self._down_block1(feat)
        feat = self._maxpool(res_feat1)
        res_feat2 = self._down_block2(feat)
        feat = self._maxpool(res_feat2)

        feat = self._mid_block(feat)

        feat = self._upsample(feat)
        feat = self._up_block2(torch.cat([feat, res_feat2], dim=1))
        feat = self._upsample(feat)
        feat = self._up_block1(torch.cat([feat, res_feat1], dim=1))

        out = self._out_block(feat)
        return out


class PosEncUNet(DDPMBase):
    """A sample DDPM model.

    NOTE: Please inherit `DDPMBase` class when you create your own DDPM model.
    """

    def __init__(self, num_channels: int, timestep_encoded_dim: int) -> None:
        super().__init__()

        self._down_block1 = PosEncConvBlock(
            ConvBlock(num_channels, 64), timestep_encoded_dim
        )
        self._down_block2 = PosEncConvBlock(ConvBlock(64, 64), timestep_encoded_dim)
        self._down_block3 = PosEncConvBlock(ConvBlock(64, 128), timestep_encoded_dim)
        self._down_block4 = PosEncConvBlock(ConvBlock(128, 128), timestep_encoded_dim)
        self._mid_block1 = PosEncConvBlock(ConvBlock(128, 256), timestep_encoded_dim)
        self._mid_block2 = PosEncConvBlock(ConvBlock(256, 256), timestep_encoded_dim)
        self._up_block4 = PosEncConvBlock(
            ConvBlock(256 + 128, 128), timestep_encoded_dim
        )
        self._up_block3 = PosEncConvBlock(ConvBlock(128, 128), timestep_encoded_dim)
        self._up_block2 = PosEncConvBlock(ConvBlock(128 + 64, 64), timestep_encoded_dim)
        self._up_block1 = PosEncConvBlock(ConvBlock(64, 64), timestep_encoded_dim)
        self._out_block = ChannelWiseConv2d(64, num_channels)

        self._maxpool = nn.MaxPool2d(2)
        self._upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self._positional_encoder = PositionalEncoder(out_dims=timestep_encoded_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self._positional_encoder.adjust_device(x)

        ts_codes = self._positional_encoder.encode(t)
        feat = x

        feat = self._down_block1(feat, ts_codes)
        feat2 = self._down_block2(feat, ts_codes)
        feat = self._maxpool(feat2)

        feat = self._down_block3(feat, ts_codes)
        feat4 = self._down_block4(feat, ts_codes)
        feat = self._maxpool(feat4)

        feat = self._mid_block1(feat, ts_codes)
        feat = self._mid_block2(feat, ts_codes)

        feat = self._upsample(feat)
        feat = self._up_block4(torch.cat([feat, feat4], dim=1), ts_codes)
        feat = self._up_block3(feat, ts_codes)

        feat = self._upsample(feat)
        feat = self._up_block2(torch.cat([feat, feat2], dim=1), ts_codes)
        feat = self._up_block1(feat, ts_codes)

        out = self._out_block(feat)
        return out
