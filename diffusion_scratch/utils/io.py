import torch
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path


class ImageIO:
    def __init__(self) -> None: ...

    @staticmethod
    def tensor_to_images(x: torch.Tensor) -> list[Image.Image]:
        converter = ToPILImage()

        def _to_pil_image(x: torch.Tensor):
            # import pdb

            # pdb.set_trace()
            # x = x * 255
            # x = x.clamp(0, 255)
            # x = x.to(torch.uint8)
            # x = x.cpu()
            image = converter(x)
            return image

        images = [_to_pil_image(img_tensor) for img_tensor in x]
        return images

    @staticmethod
    def get_fig(
        images: list[Image.Image], num_cols: int, num_rows: int, dpi: int = 1000
    ):
        fig = plt.figure(figsize=(num_cols, num_rows), dpi=dpi)
        for i in range(num_cols * num_rows):
            axes = fig.add_subplot(num_rows, num_cols, i + 1)
            axes.imshow(images[i], cmap="gray")
            axes.axis("off")
        return fig

    @staticmethod
    def save_fig(fig: Figure, save_path: Path):
        save_path = save_path.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.clf()
        plt.close()
