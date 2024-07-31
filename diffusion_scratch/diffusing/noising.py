import torch
from .utils import expand_dims


class NoiseScheduler:
    def __init__(self, num_timesteps: int, betas: torch.Tensor) -> None:
        self._num_timesteps = num_timesteps
        self._betas = betas
        self._alphas = 1 - self._betas
        self._alpha_bars = torch.cumprod(self._alphas, dim=0)

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def betas(self) -> torch.Tensor:
        return self._betas

    @property
    def alphas(self) -> torch.Tensor:
        return self._alphas

    @property
    def alpha_bars(self) -> torch.Tensor:
        return self._alpha_bars


class LinearNoiseScheduler(NoiseScheduler):
    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | None = None,
    ) -> None:
        betas = torch.linspace(
            start=beta_start, end=beta_end, steps=num_timesteps, device=device
        )
        super().__init__(num_timesteps, betas)


class DDPMManager:
    def __init__(self, scheduler: NoiseScheduler) -> None:
        self._scheduler = scheduler
        self._T = self._scheduler.num_timesteps

    def _check_timestep(self, t: torch.Tensor) -> None:
        assert (1 <= t).all() and (t <= self._T).all()


class Diffuser(DDPMManager):
    def __init__(self, scheduler: NoiseScheduler) -> None:
        super().__init__(scheduler)
        self._betas = self._scheduler.betas
        self._alphas = self._scheduler.alphas
        self._alpha_bars = self._scheduler.alpha_bars

    def add_noise_stepwise(
        self, cur_x: torch.Tensor, next_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adds noises to current images

        Parameters
        ----------
        cur_x : torch.Tensor
            A tensor of current images. Shape: `(B, C, H, W)`
        next_t : list[int]
            A tensor of next timesteps per image. Shape: `(B,)`

        Returns
        -------
        torch.Tensor
            A tensor of noised images.
        """
        self._check_timestep(next_t)
        t_index = next_t - 1
        beta = self._betas[t_index].view(*t_index.shape, 1, 1, 1)
        noise = self._create_noise_like(cur_x)
        next_x = torch.sqrt(1 - beta) * cur_x + torch.sqrt(beta) * noise
        return next_x, noise

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # import pdb

        # pdb.set_trace()

        self._check_timestep(t)
        t_index = t - 1
        alpha_bar = expand_dims(self._alpha_bars[t_index], target=x_0)
        noise = self._create_noise_like(x_0)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def _create_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x, device=x.device)
        return noise
