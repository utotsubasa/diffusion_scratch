import torch


class PositionalEncoder:
    def __init__(self, out_dims: int, device: int | str = "cpu") -> None:
        self._out_dims = out_dims
        self._device = torch.device(device)

    def adjust_device(self, sample: torch.Tensor) -> None:
        self._device = sample.device

    def encode(self, timesteps: torch.Tensor | list[int]) -> torch.Tensor:
        _timesteps = (
            timesteps
            if isinstance(timesteps, torch.Tensor)
            else torch.tensor(timesteps)
        )

        assert _timesteps.dim() == 1
        _timesteps = torch.unsqueeze(_timesteps, dim=1)

        batch_size = len(_timesteps)
        indexes = (
            torch.arange(batch_size * self._out_dims, device=self._device).view(
                batch_size, self._out_dims
            )
            % self._out_dims
        )
        rad_denos = 10000 ** (indexes / self._out_dims)
        rads = _timesteps / rad_denos

        codes = torch.zeros_like(rads)
        codes[:, 0::2] = torch.sin(rads[:, 0::2])
        codes[:, 1::2] = torch.cos(rads[:, 1::2])
        return codes

    @property
    def device(self) -> torch.device:
        return self._device
