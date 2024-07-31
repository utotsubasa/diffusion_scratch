import torch
from typing import overload
import torch.nn as nn
from contextlib import contextmanager


@contextmanager
def switch_phase(model: nn.Module):
    orig_phase = model.training
    model.train(mode=not orig_phase)
    yield
    model.train(mode=orig_phase)


@overload
def expand_dims(
    x: torch.Tensor, target: None = None, num_extra_dims: int = 0
) -> torch.Tensor: ...
@overload
def expand_dims(
    x: torch.Tensor, target: torch.Tensor, num_extra_dims: None = None
) -> torch.Tensor: ...
def expand_dims(
    x: torch.Tensor,
    target: torch.Tensor | None = None,
    num_extra_dims: int | None = None,
) -> torch.Tensor:
    # import pdb

    # pdb.set_trace()

    if target is None:
        assert num_extra_dims is not None
    else:
        assert num_extra_dims is None

        assert x.shape[0] == target.shape[0]
        num_extra_dims = target.dim() - 1

    extra_shape = [1 for _ in range(num_extra_dims)]
    return x.view(*x.shape, *extra_shape)
