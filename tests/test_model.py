import pytest
from diffusion_scratch.models.unet import UNet
import torch

@pytest.mark.parametrize(["num_channels"], [(1,), (3,)])
def test_unet(num_channels: int):
    net = UNet(num_channels=num_channels)
    size = (16, num_channels, 64, 64)
    dummy_input = torch.rand(size)
    output: torch.Tensor = net(dummy_input)
    assert output.shape == size
