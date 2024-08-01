from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

__all__ = ["DDPMBase"]


class DDPMBase(nn.Module, metaclass=ABCMeta):
    """A base class of DDPM model"""

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            A tensor of images. Shape: `(B, C, H, W)`
        t : torch.Tensor
            A tensor of timesteps. Shape: `(B,)`

        Returns
        -------
        torch.Tensor
            A tensor of noises added on images
            from the initial timestep to the given timesteps.
        """
        ...
