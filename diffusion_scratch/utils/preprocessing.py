import torchvision.transforms as tvt
from pathlib import Path
from PIL import Image
import torch
from typing import Any
from abc import ABCMeta, abstractmethod


class TransformModule(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any: ...


class LoadImage(TransformModule):
    def __call__(self, image_path: Path) -> Image.Image:
        image = Image.open(image_path)
        return image


class Resize(TransformModule):
    def __init__(self, size: tuple[int, ...]) -> None:
        super().__init__()
        self._converter = tvt.Resize(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = self._converter(image)
        return image


class ToTensor(TransformModule):
    def __init__(self) -> None:
        super().__init__()
        self._converter = tvt.ToTensor()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor = self._converter(image)
        return tensor


class Transforms:
    def __init__(self, *modules: TransformModule) -> None:
        self._modules = modules

    def __call__(self, data: Any) -> torch.Tensor:
        out = data
        for module in self._modules:
            out = module(out)

        return out
