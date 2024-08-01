import torch
import torch.utils.data.dataset as torch_dataset
from .preprocessing import Transforms
from pathlib import Path
from torchvision.datasets import MNIST, CIFAR10


class BaseDataset(torch_dataset.Dataset[torch.Tensor]):
    """A base class of Dataset"""

    ...


class Dataset(BaseDataset):
    """A subclass of BaseDataset for a basic usage"""

    def __init__(self, img_paths: list[Path], transforms: Transforms) -> None:
        """
        Parameters
        ----------
        img_paths : list[Path]
            A list of paths to image
        transforms : Transforms
            A module for image preprocessing.
        """
        super().__init__()
        self._img_paths = img_paths
        self._transforms = transforms

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self._img_paths[index]
        img = self._transforms(img_path)
        return img

    def __len__(self) -> int:
        return len(self._img_paths)


class MNISTDataset(BaseDataset):
    """A subclass of BaseDataset for MNIST"""

    def __init__(self, transforms: Transforms) -> None:
        super().__init__()
        self._mnist = MNIST(root="../data/mnist", train=True, download=True)
        self._transforms = transforms

    def __getitem__(self, index: int) -> torch.Tensor:
        data: tuple[torch.Tensor, int] = self._mnist[index]
        image, _ = data
        image = self._transforms(image)
        return image

    def __len__(self) -> int:
        return len(self._mnist)


class Cifar10Dataset(BaseDataset):
    """A subclass of BaseDataset for Cifar10"""

    def __init__(self, transforms: Transforms) -> None:
        super().__init__()
        self._cifar = CIFAR10(root="../data/cifar", train=True, download=True)
        self._transforms = transforms

    def __getitem__(self, index: int) -> torch.Tensor:
        data: tuple[torch.Tensor, int] = self._cifar[index]
        image, _ = data
        image = self._transforms(image)
        return image

    def __len__(self) -> int:
        return len(self._cifar)
