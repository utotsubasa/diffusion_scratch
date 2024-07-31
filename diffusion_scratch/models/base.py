from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn

__all__ = ["DDPMBase"]


class DDPMBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
