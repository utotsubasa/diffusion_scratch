import torch
import math


class PositionalEncoder:
    """A class for encoding timesteps"""

    def __init__(self, out_dims: int, device: int | str = "cpu") -> None:
        """
        Parameters
        ----------
        out_dims : int
            A dimension size of timestep encoding.
        device : int | str, optional
            A device, by default "cpu"
        """
        self._out_dims = out_dims
        self._device = torch.device(device)

    def adjust_device(self, sample: torch.Tensor) -> None:
        self._device = sample.device

    def encode(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Encodes given timesteps

        Parameters
        ----------
        timesteps : torch.Tensor
            A tensor of timesteps. Shape: `(B,)`

        Returns
        -------
        torch.Tensor
            A tensor of encoded timesteps. Shape: `(B, out_dims)`
        """
        assert timesteps.dim() == 1
        _timesteps = torch.unsqueeze(timesteps, dim=1)

        batch_size = len(_timesteps)
        indexes = (
            torch.arange(batch_size * self._out_dims, device=self._device).view(
                batch_size, self._out_dims
            )
            % self._out_dims
        )
        # rad_denos = 10000 ** (indexes / self._out_dims)
        rad_denos = torch.exp(math.log(10000) * indexes / self._out_dims)

        rads = _timesteps / rad_denos

        codes = torch.zeros_like(rads)
        codes[:, 0::2] = torch.sin(rads[:, 0::2])
        codes[:, 1::2] = torch.cos(rads[:, 1::2])
        return codes

    @property
    def device(self) -> torch.device:
        return self._device
