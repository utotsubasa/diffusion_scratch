import pytest
from pytest import FixtureRequest
from diffusion_scratch.models import UNet, PosEncUNet, PositionalEncoder

import torch


@pytest.mark.parametrize(["num_channels"], [(1,), (3,)])
def test_unet(num_channels: int):
    net = UNet(num_channels=num_channels)
    size = (16, num_channels, 64, 64)
    dummy_input = torch.rand(size)
    output: torch.Tensor = net(dummy_input)
    assert output.shape == size


class TestUNet:
    @pytest.fixture(params=(1, 3))
    def num_channels(self, request: FixtureRequest):
        num_channels = request.param
        return num_channels

    @pytest.fixture
    def unet(self, num_channels: int):
        net = UNet(num_channels)
        return net

    def test_output(self, unet: UNet, num_channels: int):
        size = (16, num_channels, 64, 64)
        dummy = torch.rand(size)
        output: torch.Tensor = unet(dummy)
        assert output.shape == size


class TestPosEncUNet:
    @pytest.fixture(params=(1, 3))
    def num_channels(self, request: FixtureRequest):
        num_channels = request.param
        return num_channels

    @pytest.fixture(params=(20,))
    def timestep_encoded_dim(self, request: FixtureRequest):
        dim = request.param
        return dim

    @pytest.fixture
    def pos_enc_unet(self, num_channels: int, timestep_encoded_dim: int):
        unet = PosEncUNet(
            num_channels=num_channels, timestep_encoded_dim=timestep_encoded_dim
        )
        return unet

    def test_output(self, num_channels: int, pos_enc_unet: PosEncUNet):
        batch_size = 32
        dummy_x = torch.rand((batch_size, num_channels, 32, 32))
        dummy_t = torch.randint(low=0, high=30, size=(batch_size,))
        output: torch.Tensor = pos_enc_unet(dummy_x, dummy_t)
        assert output.shape == dummy_x.shape


class TestPositionalEncoder:
    CODE_OUT_DIMS: int = 10

    @pytest.fixture
    def encoder(self):
        encoder = PositionalEncoder(out_dims=self.CODE_OUT_DIMS)
        return encoder

    def test_adjust_device(self, encoder: PositionalEncoder):
        cpu = torch.device("cpu")
        cuda = torch.device("cuda:0")
        dummy = torch.tensor(0, device=cuda)
        assert encoder.device == cpu
        encoder.adjust_device(dummy)
        assert encoder.device == cuda

    def test_positional_encoder(self, encoder: PositionalEncoder):
        timesteps = torch.tensor([1, 2, 3])
        codes = encoder.encode(timesteps)
        assert codes.shape == (len(timesteps), self.CODE_OUT_DIMS)
