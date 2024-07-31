from diffusion_scratch.models import PosEncUNet
from diffusion_scratch.utils.preprocessing import (
    LoadImage,
    ToTensor,
    Transforms,
    Resize,
)
from diffusion_scratch.utils.datasets import Dataset, MNISTDataset, Cifar10Dataset
from diffusion_scratch.utils.data_loader import DataLoader
from pathlib import Path
from diffusion_scratch.diffusing.noising import Diffuser, LinearNoiseScheduler
from torch.nn.modules.loss import MSELoss
from diffusion_scratch.training import TrainSettings, TrainingInfo, Trainer
import torch
import torch.optim as optim
from diffusion_scratch.diffusing.sampling import Sampler
from omegaconf import OmegaConf
import argparse
from typing import Literal
from datetime import datetime
from zoneinfo import ZoneInfo


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=Path)
    return parser


def get_args() -> argparse.Namespace:
    parser = _get_parser()
    args = parser.parse_args()
    return args


def _get_image_channel_size(image_type: Literal["rgb", "gray"]):
    match image_type:
        case "rgb":
            return 3
        case "gray":
            return 1


def _get_dataset(
    dataset_type: Literal["mnist", "cifar10", "celeba"],
    resize: tuple[int, int] | None = None,
    dataset_root: str | None = None,
):
    modules = [ToTensor()]
    if resize is not None:
        modules = [Resize(size=resize)] + modules
    match dataset_type:
        case "mnist":
            transforms = Transforms(*modules)
            dataset = MNISTDataset(transforms=transforms)
        case "cifar10":
            transforms = Transforms(*modules)
            dataset = Cifar10Dataset(transforms=transforms)
        case "celeba":
            assert dataset_root is not None
            img_paths = list(Path(dataset_root).iterdir())

            modules = [LoadImage()] + modules
            transforms = Transforms(*modules)
            dataset = Dataset(img_paths, transforms=transforms)
    return dataset


def main():
    args = get_args()
    config_path: Path = args.config_path
    config_path = config_path.resolve()

    config = OmegaConf.load(config_path)

    train_settings = TrainSettings(
        num_epochs=config.num_epochs,
        num_timesteps=config.num_timesteps,
        main_device=torch.device(config.main_device),
        val_freq=config.val_freq,
    )

    log_root = Path("../outputs")
    date = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y_%m_%d_%H_%M_%S")
    train_info = TrainingInfo(
        log_root=log_root, dataset_type=config.dataset_type, date=date
    )

    num_channels = _get_image_channel_size(config.image_type)
    timestep_encoded_dim: int = config.timestep_encoded_dim
    model = PosEncUNet(num_channels, timestep_encoded_dim)

    resize_hw: tuple[int, int] | None = OmegaConf.select(config, "resize_hw")
    dataset_root: str | None = OmegaConf.select(config, "dataset_root")
    dataset = _get_dataset(
        config.dataset_type, resize=resize_hw, dataset_root=dataset_root
    )
    data_loader = DataLoader(
        dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )
    noise_scheduler = LinearNoiseScheduler(
        num_timesteps=train_settings.num_timesteps, device=train_settings.main_device
    )
    diffuser = Diffuser(noise_scheduler)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss = MSELoss()

    output_hw: tuple[int, int] = (
        resize_hw
        if resize_hw is not None
        else (dataset[0].shape[1], dataset[0].shape[2])
    )
    sampler = Sampler(
        noise_scheduler,
        model,
        (8, num_channels, *output_hw),
        train_settings.main_device,
    )
    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        diffuser=diffuser,
        optimizer=optimizer,
        loss=loss,
        settings=train_settings,
        info=train_info,
        sampler=sampler,
    )
    trainer()


if __name__ == "__main__":
    main()
