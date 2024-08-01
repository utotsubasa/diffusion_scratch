from .noising import DDPMManager, NoiseScheduler
from diffusion_scratch.models.base import DDPMBase
import torch
from .utils import expand_dims, switch_phase
from tqdm import tqdm


class Sampler(DDPMManager):
    """A subclass of DDPMManager which manages backward/sampling process."""

    def __init__(
        self,
        scheduler: NoiseScheduler,
        ddpm: DDPMBase,
        size: tuple[int, ...],
        device: torch.device,
    ) -> None:
        """
        Parameters
        ----------
        scheduler : NoiseScheduler
            A noise scheduler module
        ddpm : DDPMBase
            A target DDPM model
        size : tuple[int, ...]
            A size of image to be created
        device : torch.device
            A device
        """
        super().__init__(scheduler)
        self._alphas = self._scheduler.alphas
        self._alpha_bars = self._scheduler.alpha_bars
        self._net = ddpm
        self._size = size
        self._device = device

    def denoise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reduces noises from given images just by one step

        Parameters
        ----------
        x : torch.Tensor
            A tensor of target images. Shape: `(B, C, H, W)`
        t : torch.Tensor
            A tensor of target timesteps. Shape: `(B,)`

        Returns
        -------
        torch.Tensor
            A tensor of images where noises are reduced.
        """
        self._check_timestep(t)
        t_index = t - 1
        alpha = expand_dims(self._alphas[t_index], target=x)
        alpha_bar = expand_dims(self._alpha_bars[t_index], target=x)
        alpha_bar_prev = expand_dims(self._alpha_bars[t_index - 1], target=x)

        with switch_phase(self._net):
            with torch.no_grad():
                out: torch.Tensor = self._net(x, t)

        out_coef = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        mu_coef = 1 / torch.sqrt(alpha)
        mu = mu_coef * (x - out_coef * out)

        std = torch.sqrt((1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar))
        noise = self._generate_noise(x, t)

        next_x = mu + std * noise

        return next_x

    def sample(self) -> torch.Tensor:
        """Samples images from a given distribution.

        Returns
        -------
        torch.Tensor
            A tensor of sampled images
        """
        x = torch.randn(self._size, device=self._device)

        for i in tqdm(range(self._T, 0, -1), desc="sampling"):
            t = torch.tensor(
                [i for _ in range(self._size[0])], device=self._device, dtype=torch.long
            )
            x = self.denoise(x, t)

        return x

    def _generate_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generates noises from a given distribution.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of images as which the generated noises will be the same.
        t : torch.Tensor
            A tensor of timesteps.

        Returns
        -------
        torch.Tensor
            A tensor of noises of the given timesteps
        """

        if t == 1:
            noise = torch.zeros_like(x, device=x.device, dtype=torch.float)
        else:
            noise = torch.randn_like(x, device=x.device)
        return noise
