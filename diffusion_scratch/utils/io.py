import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path


class ImageIO:
    """A class to input/output images."""

    def __init__(self) -> None: ...

    @staticmethod
    def tensor_to_images(x: torch.Tensor) -> list[Image.Image]:
        """Converts images from tensor to PIL

        Parameters
        ----------
        x : torch.Tensor
            A tensor of images

        Returns
        -------
        list[Image.Image]
            A list of images of PIL format
        """
        converter = ToPILImage()

        def _to_pil_image(x: torch.Tensor):
            image = converter(x)
            return image

        images = [_to_pil_image(img_tensor) for img_tensor in x]
        return images

    @staticmethod
    def get_fig(
        images: list[Image.Image], num_cols: int, num_rows: int, dpi: int = 1000
    ):
        """Arranges images.

        Parameters
        ----------
        images : list[Image.Image]
            A list of images of PIL format
        num_cols : int
            The number of columns
        num_rows : int
            The number of rows
        dpi : int, optional
            A dpi of a created figure, by default 1000

        Returns
        -------
        _type_
            A figure where images are arranged
        """
        fig = plt.figure(figsize=(num_cols, num_rows), dpi=dpi)
        for i in range(num_cols * num_rows):
            axes = fig.add_subplot(num_rows, num_cols, i + 1)
            axes.imshow(images[i], cmap="gray")
            axes.axis("off")
        return fig

    @staticmethod
    def save_fig(fig: Figure, save_path: Path):
        """Saves figures.

        Parameters
        ----------
        fig : Figure
            A figure to be saved
        save_path : Path
            A path where the figure is saved
        """
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.clf()
        plt.close()
