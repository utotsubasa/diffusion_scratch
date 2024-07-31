from torch.nn.modules.loss import (
    _Loss as TorchLoss,  # pyright: ignore[reportPrivateUsage]
)

__all__ = ["Loss"]


class Loss(TorchLoss): ...
