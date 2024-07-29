import torch
from torch import nn
from typing import Any

class ChannelWiseConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__(*(in_channels, out_channels, 1), **kwargs) # type: ignore

class CBRLayer(nn.Module):
    """A Layer which consists of Conv, BatchNormalizaiton and Relu."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        bn = nn.BatchNorm2d(num_features=out_channels)
        relu = nn.ReLU()
        self._net = nn.Sequential(conv, bn, relu)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 2) -> None:
        super().__init__()
        
        net_list: list[nn.Module] = []
        net_list.append(CBRLayer(in_channels=in_channels, out_channels=out_channels))
        for _ in range(num_layers - 1):
            net_list.append(CBRLayer(in_channels=out_channels, out_channels=out_channels))
        self._net = nn.Sequential(*net_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._net(x)
        return out

class UNet(nn.Module):
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
        
        
