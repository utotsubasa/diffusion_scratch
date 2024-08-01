from diffusion_scratch.utils.data_loader import DataLoader
from diffusion_scratch.models.base import DDPMBase
from dataclasses import dataclass
from torch.optim import Optimizer
import torch
from torch.nn.modules.loss import _Loss as Loss  # type: ignore
from diffusion_scratch.diffusing.noising import Diffuser
from diffusion_scratch.diffusing.sampling import Sampler
from rich.progress import track
from diffusion_scratch.utils.io import ImageIO
from pathlib import Path
from torch.nn import DataParallel


@dataclass
class TrainSettings:
    num_epochs: int
    num_timesteps: int
    main_device: torch.device
    multi_gpus: list[int] | None = None
    val_freq: int = 10


@dataclass
class TrainingInfo:
    log_root: Path
    dataset_type: str
    date: str


class Trainer:
    """A class for training DDPM model"""

    def __init__(
        self,
        model: DDPMBase,
        data_loader: DataLoader,
        diffuser: Diffuser,
        optimizer: Optimizer,
        loss: Loss,
        settings: TrainSettings,
        info: TrainingInfo,
        sampler: Sampler | None = None,
    ) -> None:

        self._model = model
        self._data_loader = data_loader
        self._diffuser = diffuser
        self._sampler = sampler
        self._optimizer = optimizer
        self._loss = loss
        self._settings = settings
        self._info = info

        self._check_train_settings()

        if self._settings.multi_gpus is not None:
            self._model = DataParallel(
                self._model, device_ids=self._settings.multi_gpus
            )

    def __call__(self) -> None:
        """Executes training"""

        self._model.to(self._settings.main_device)
        train_results: list[float] = []

        for epoch in range(self._settings.num_epochs):
            if epoch > 0:
                train_result = self._train_one_epoch(epoch)
                train_results.append(train_result)

            if self._enter_val_phase(epoch):
                self._evaluate(epoch)

    def _evaluate(self, epoch: int) -> None:
        """Evaluates image

        Parameters
        ----------
        epoch : int
            Current epoch
        """
        assert self._sampler is not None

        # sample images in a form of a tensor
        x_sampled = self._sampler.sample()

        # convert images from a tensor to the PIL format
        images = ImageIO.tensor_to_images(x_sampled)

        # arrange images
        fig = ImageIO.get_fig(images, 4, 2)

        # save images
        log_dir = self._info.log_root / self._info.dataset_type / self._info.date
        image_filename = f"image_{epoch}.png"
        ImageIO.save_fig(fig, log_dir / image_filename)

    def _train_one_epoch(self, epoch: int) -> float:
        loss_sum: float = 0.0
        total_num_data: int = 0

        for inputs in track(
            self._data_loader, description=f"epoch {epoch}", disable=False
        ):
            score, _ = self._train_one_batch(inputs)
            loss_sum += score
            total_num_data += 1

        epoch_score = loss_sum / total_num_data
        print(f"loss: {epoch_score}")
        return epoch_score

    def _train_one_batch(self, inputs: torch.Tensor):
        self._model.train()
        self._optimizer.zero_grad()

        x = inputs.to(device=self._settings.main_device)

        # sample timesteps at randam from a uniform distribution.
        t = torch.randint(
            low=1,
            high=self._settings.num_timesteps + 1,
            size=(len(x),),
            device=self._settings.main_device,
        )

        # add noise to images
        x_noised, noise = self._diffuser.add_noise(x, t)

        # predicts added noises from noisy images
        noise_pred = self._model(x_noised, t)

        # compare added noises and predicted noises
        loss: torch.Tensor = self._loss(noise, noise_pred)

        loss.backward()
        self._optimizer.step()

        loss_score = loss.sum().item()
        num_data = len(x)
        return loss_score, num_data

    def _check_train_settings(self): ...

    def _enter_val_phase(self, epoch: int) -> bool:
        if self._sampler is None:
            return False

        flag = epoch == 0
        flag = flag or (epoch + 1) % self._settings.val_freq == 0
        flag = flag or (epoch + 1) == self._settings.num_epochs
        return flag
