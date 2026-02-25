import functools
import math
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, override

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, Dataset

from .data import AudioStreamingDataLoader, DynamicMixingDataset
from .losses import BaseLoss
from .metrics import BaseMetric
from .models import ADM, NAC, ADDSERQDiT, SGMSEUNet
from .stft import STFT


@dataclass
class LogConfig:
    """Configuration for logging losses and metrics."""

    on_train_step: bool = False
    on_train_epoch: bool = True
    on_val_step: bool = False
    on_val_epoch: bool = True
    on_test_step: bool = False
    on_test_epoch: bool = True


class BaseLightningModule(L.LightningModule):
    """Base class for Lightning modules."""

    val_metrics: Mapping[str, BaseMetric] | None
    test_metrics: Mapping[str, BaseMetric] | None
    log_cfg: LogConfig
    debug_sample: tuple[int, int] | None

    @abstractmethod
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        """Training, validation, or test step.

        Args:
            batch: A batch from the dataloader.
            stage: `"train"`, `"val"`, or `"test"`.
            batch_idx: Index of the batch.
            metrics: Metrics to compute. `None` if `stage` is `"train"` or if no metrics are defined.

        Returns:
            Tuple of loss dictionary, metrics dictionary, and debug samples dictionary. Each debug sample must have
            shape `(batch_size, num_channels, num_samples)`.
        """

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Training step.

        Args:
            batch: A batch from the training dataloader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary with losses.
        """
        loss, metrics, _ = self.step(batch, "train", batch_idx)
        self.log_metrics(loss, metrics, "train", self.log_cfg.on_train_step, self.log_cfg.on_train_epoch)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Validation step.

        Args:
            batch: A batch from the validation dataloader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary with losses.
        """
        loss, metrics, debug_samples = self.step(batch, "val", batch_idx, self.val_metrics)
        self.log_debug_samples(batch, batch_idx, debug_samples)
        self.log_metrics(loss, metrics, "val", self.log_cfg.on_val_step, self.log_cfg.on_val_epoch)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> dict[str, Tensor]:
        """Test step.

        Args:
            batch: A batch from the test dataloader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary with losses.
        """
        loss, metrics, _ = self.step(batch, "test", batch_idx, self.test_metrics)
        self.log_metrics(loss, metrics, "test", self.log_cfg.on_test_step, self.log_cfg.on_test_epoch)
        return loss

    def log_metrics(
        self, loss: dict[str, Tensor], metrics: dict[str, float], stage: str, on_step: bool, on_epoch: bool
    ) -> None:
        """Log losses and metrics."""
        for key, value in {**loss, **metrics}.items():
            assert isinstance(value, Tensor | float)
            self.log(f"{stage}_{key}", value, on_step=on_step, on_epoch=on_epoch)

    def log_debug_samples(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int, debug_samples: dict[str, Tensor]
    ) -> None:
        """Log debug audio samples to W&B."""
        wandb_logger = next((logger for logger in self.loggers if isinstance(logger, WandbLogger)), None)
        if wandb_logger is None or self.debug_sample is None or batch_idx != self.debug_sample[0]:
            return
        for name, x in debug_samples.items():
            x_cpu = x[self.debug_sample[1], 0, :].cpu().float().numpy()
            fs = batch[2][self.debug_sample[1]].item()
            wandb_logger.log_audio(
                key=name,
                audios=[x_cpu / max(abs(x_cpu))],
                step=self.global_step,
                sample_rate=[fs],
            )


class ConfigureOptimizersMixin(L.LightningModule):
    """Mixin for standard configuration of optimizer and learning rate scheduler."""

    optimizer: Callable[[Iterator[nn.Parameter]], Optimizer]
    lr_scheduler: Mapping[str, Any] | None

    def configure_optimizers(self) -> Any:
        """Configure optimizers.

        Returns:
            Dictionary with optimizer, learning rate scheduler, and learning rate scheduler configuration.
        """
        optimizer = self.optimizer(self.parameters())
        output: dict[str, Any] = {"optimizer": optimizer}
        if self.lr_scheduler is not None:
            output["lr_scheduler"] = {k: v(optimizer) if k == "scheduler" else v for k, v in self.lr_scheduler.items()}
        return output


class EDMMixin(L.LightningModule):
    """Mixin for training and sampling as in EDM."""

    model: nn.Module
    num_steps: int
    norm_factor: float
    sigma_data: float
    p_mean: float
    p_sigma: float
    s_churn: float
    s_min: float
    s_max: float
    s_noise: float
    sigma_min: float
    sigma_max: float
    rho: float

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the loss as in EDM."""
        log_sigma = self.p_mean + self.p_sigma * torch.randn(y.shape[0], dtype=y.real.dtype, device=y.device)
        sigma = log_sigma.exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = sigma.view(-1, *(1,) * (y.ndim - 1)) * torch.randn_like(y)
        loss = weight.view(-1, *(1,) * (y.ndim - 1)) * (self.denoiser(y + noise, x, sigma) - y).abs().pow(2)
        return loss.mean()

    def denoiser(self, y: Tensor, x: Tensor, sigma: Tensor) -> Tensor:
        """Compute the denoiser parametrization as in EDM."""
        sigma_broad = sigma.view(-1, *(1,) * (y.ndim - 1))
        c_skip = self.sigma_data**2 / (sigma_broad**2 + self.sigma_data**2)
        c_out = self.sigma_data * sigma_broad / (self.sigma_data**2 + sigma_broad**2).sqrt()
        c_in_y = 1 / (self.sigma_data**2 + sigma_broad**2).sqrt()
        c_in_x = 1 / self.sigma_data
        c_noise = 0.25 * sigma.log()
        return c_skip * y + c_out * self.model(c_in_y * y, c_in_x * x, c_noise)

    @torch.no_grad()
    def solve(self, x: Tensor, num_steps: int) -> Tensor:
        """Sample using the Heun method as in EDM."""
        assert not self.model.training, "Model must be in eval mode to sample."
        t = torch.tensor(
            [self.sampling_step(i) if i < num_steps else 0.0 for i in range(num_steps + 1)],
            device=x.device,
            dtype=x.real.dtype,
        )
        y = t[0] * torch.randn_like(x)
        for i in range(num_steps):
            if self.s_churn > 0 and self.s_min <= t[i] <= self.s_max:
                gamma = min(self.s_churn / num_steps, math.sqrt(2) - 1)
                t_hat = t[i] * (1 + gamma)
                y_hat = y + (t_hat**2 - t[i] ** 2).sqrt() * torch.randn_like(y) * self.s_noise
            else:
                t_hat = t[i]
                y_hat = y
            d = (y_hat - self.denoiser(y_hat, x, t_hat[None])) / t_hat
            y = y_hat + (t[i + 1] - t_hat) * d
            if i < num_steps - 1:
                d_next = (y - self.denoiser(y, x, t[i + 1, None])) / t[i + 1]
                y = y_hat + 0.5 * (t[i + 1] - t_hat) * (d + d_next)
        return y

    def sampling_step(self, i: int) -> float:
        """Compute the i-th sampling step."""
        return (
            self.sigma_max ** (1 / self.rho)
            + i / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho


class LightningModule(BaseLightningModule, ConfigureOptimizersMixin):
    """Simple Lightning module for training models to directly predict clean speech given noisy speech."""

    def __init__(
        self,
        model: nn.Module,
        loss: BaseLoss,
        optimizer: Callable[[Iterator[nn.Parameter]], Optimizer] = Adam,
        lr_scheduler: Mapping[str, Any] | None = None,
        val_metrics: Mapping[str, BaseMetric] | None = None,
        test_metrics: Mapping[str, BaseMetric] | None = None,
        log_cfg: LogConfig | None = None,
        debug_sample: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the simple Lightning module.

        Args:
            model: Model to train.
            loss: Loss module.
            optimizer: Optimizer constructor.
            lr_scheduler: Learning rate scheduler configuration.
            val_metrics: Metrics to compute during validation.
            test_metrics: Metrics to compute during testing.
            log_cfg: Logging configuration.
            debug_sample: Tuple `(batch_idx, sample_idx)` to log debug audio samples to W&B during validation.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = debug_sample

    @override
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        x, y, _ = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        metric_vals = compute_metrics(y_hat, y, metrics)
        debug_samples = {"input": x, "reference": y, "output": y_hat} if self.current_epoch == 0 else {"output": y_hat}
        return loss, metric_vals, debug_samples

    def forward(self, x: Tensor) -> Tensor:
        """Enhance the input audio."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        return self.model(x)


class SGMSELightningModule(BaseLightningModule, ConfigureOptimizersMixin):
    """SGMSE Lightning module."""

    def __init__(
        self,
        model: SGMSEUNet,
        stft: STFT,
        num_steps: int = 30,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        gamma: float = 1.5,
        t_eps: float = 0.03,
        corrector_snr: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.15,
        optimizer: Callable[[Iterator[nn.Parameter]], Optimizer] = Adam,
        lr_scheduler: Mapping[str, Any] | None = None,
        val_metrics: Mapping[str, BaseMetric] | None = None,
        test_metrics: Mapping[str, BaseMetric] | None = None,
        log_cfg: LogConfig | None = None,
        debug_sample: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the SGMSE Lightning module."""
        super().__init__()
        self.model = model
        self.stft = stft
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.gamma = gamma
        self.t_eps = t_eps
        self.corrector_snr = corrector_snr
        self.alpha = alpha
        self.beta = beta
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = debug_sample

    def transform(self, x: Tensor) -> Tensor:
        """Compute the STFT, compress, and scale."""
        x = self.stft(x)
        return torch.polar(self.beta * x.abs() ** self.alpha, x.angle())

    def inverse_transform(self, x: Tensor, n: int) -> Tensor:
        """Decompress, descale, and compute the inverse STFT."""
        x = torch.polar((x.abs() / self.beta) ** (1 / self.alpha), x.angle())
        return self.stft.inverse(x, n=n)

    @override
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        x, y, _ = batch
        # peak-normalize
        factor = x.abs().amax(dim=(1, 2), keepdim=True)
        x_norm, y_norm = x / factor, y / factor
        x_stft, y_stft = self.transform(x_norm), self.transform(y_norm)
        loss = {"loss": self.loss(x_stft, y_stft)}
        if metrics or stage == "val" and self.debug_sample is not None and batch_idx == self.debug_sample[0]:
            y_hat_stft = self.solve(x_stft, self.num_steps)
            y_hat_norm = self.inverse_transform(y_hat_stft, n=x.shape[-1])
            # undo peak-normalization
            y_hat = y_hat_norm * factor
            metric_vals = compute_metrics(y_hat, y, metrics)
            debug = {"input": x, "reference": y, "output": y_hat} if self.current_epoch == 0 else {"output": y_hat}
            return loss, metric_vals, debug
        return loss, {}, {}

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the loss."""
        t = torch.rand(y.shape[0], dtype=y.real.dtype, device=y.device) * (1 - self.t_eps) + self.t_eps
        sigma = self.sigma(t).reshape(-1, 1, 1, 1)
        z = torch.randn_like(y)
        y_t = (-self.gamma * t).exp().reshape(-1, 1, 1, 1) * (y - x) + x + sigma * z  # eq. (30-32)
        return (sigma * self.score(x, y_t, t) + z).abs().pow(2).mean()  # eq. (33)

    def sigma(self, t: Tensor) -> Tensor:
        """Noise schedule."""
        sigma_ratio = self.sigma_max / self.sigma_min
        num = sigma_ratio ** (2 * t) - (-2 * self.gamma * t).exp()
        den = 1 + self.gamma / math.log(sigma_ratio)
        return self.sigma_min * (num / den).sqrt()

    def score(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        """Estimate the score function."""
        return -self.model(x, y, t.log()) / t.reshape(-1, 1, 1, 1)  # eq. (34)

    @torch.no_grad()
    def solve(self, x: Tensor, num_steps: int) -> Tensor:
        """Sample using the predictor-corrector method."""
        assert not self.model.training, "Model must be in eval mode to sample."
        t = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device, dtype=x.real.dtype)
        sigma = self.sigma(t)
        eps = 2 * (self.corrector_snr * sigma) ** 2
        y = x + sigma[0] * torch.randn_like(x)
        sigma_ratio = self.sigma_max / self.sigma_min
        for i in range(num_steps):
            # corrector step
            y += eps[i] * self.score(x, y, t[i, None]) + (2 * eps[i]) ** 0.5 * torch.randn_like(y)
            # predictor step
            g = self.sigma_min * sigma_ratio ** t[i] * (2 * math.log(sigma_ratio)) ** 0.5
            if i < num_steps - 1:  # reverse SDE step
                y_dot = self.gamma * (x - y) - g**2 * self.score(x, y, t[i, None])
                y += (t[i + 1] - t[i]) * y_dot + g * (t[i] - t[i + 1]).sqrt() * torch.randn_like(y)
            else:  # probability flow ODE step
                y_dot = self.gamma * (x - y) - g**2 * self.score(x, y, t[i, None]) * 0.5
                y += (t[i + 1] - t[i]) * y_dot
        return y

    def forward(self, x: Tensor, num_steps: int | None = None) -> Tensor:
        """Enhance the input audio."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        # peak-normalize
        factor = x.abs().amax(dim=(1, 2), keepdim=True)
        x = x / factor
        x_stft = self.transform(x)
        # pad to multiple of model.downsampling_factor
        n_pad = (self.model.downsampling_factor - x_stft.shape[-1]) % self.model.downsampling_factor
        x_stft_pad = F.pad(x_stft, (0, n_pad))
        y_stft_pad = self.solve(x_stft_pad, self.num_steps if num_steps is None else num_steps)
        y_stft = y_stft_pad[..., : y_stft_pad.shape[-1] - n_pad]
        y = self.inverse_transform(y_stft, n=x.shape[-1])
        # undo peak-normalization
        return y * factor


class NACLightningModule(BaseLightningModule):
    """Lightning module for neural audio codec."""

    def __init__(
        self,
        generator: NAC,
        discriminator: nn.Module | Iterable[nn.Module],
        reconstruction_loss: BaseLoss,
        adversarial_loss_weight: float,
        feature_loss_weight: float,
        reconstruction_loss_weight: float,
        codebook_loss_weight: float,
        commitment_loss_weight: float,
        generator_optimizer: Callable[[Iterator[nn.Parameter]], Optimizer],
        discriminator_optimizer: Callable[[Iterator[nn.Parameter]], Optimizer],
        generator_grad_clip: float = 0.0,
        discriminator_grad_clip: float = 0.0,
        val_metrics: Mapping[str, BaseMetric] | None = None,
        test_metrics: Mapping[str, BaseMetric] | None = None,
        log_cfg: LogConfig | None = None,
        debug_sample: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the neural audio codec Lightning module."""
        super().__init__()
        self.generator = generator
        self.discriminator = nn.ModuleList([discriminator] if isinstance(discriminator, nn.Module) else discriminator)
        self.reconstruction_loss = reconstruction_loss
        self.adversarial_loss_weight = adversarial_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.codebook_loss_weight = codebook_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_grad_clip = generator_grad_clip
        self.discriminator_grad_clip = discriminator_grad_clip
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = debug_sample
        self.automatic_optimization = False

    def discriminator_forward(self, x: Tensor) -> tuple[list[Tensor], list[list[Tensor]]]:
        """Forward pass through all discriminators."""
        all_outputs: list[Tensor] = []
        all_featuress: list[list[Tensor]] = []
        for discriminator in self.discriminator:
            outputs, featuress = discriminator(x)
            all_outputs.extend(outputs)
            all_featuress.extend(featuress)
        return all_outputs, all_featuress

    def discriminator_step(self, x: Tensor, y: Tensor) -> Tensor:
        """Discriminator step."""
        real, fake = x.detach(), y.detach()
        real_preds, _ = self.discriminator_forward(real)
        fake_preds, _ = self.discriminator_forward(fake)
        loss: Tensor | float = 0.0
        for real_pred, fake_pred in zip(real_preds, fake_preds):
            loss += F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()
        assert isinstance(loss, Tensor)
        return loss / len(real_preds)

    def generator_step(self, x: Tensor, y: Tensor, codebook_loss: Tensor, commit_loss: Tensor) -> dict[str, Tensor]:
        """Generator step."""
        _, real_featss = self.discriminator_forward(x)
        fake_preds, fake_featss = self.discriminator_forward(y)
        adv_loss: Tensor | float = 0.0
        feat_loss: Tensor | float = 0.0
        for fake_pred, fake_feats, real_feats in zip(fake_preds, fake_featss, real_featss):
            adv_loss += -fake_pred.mean()
            for real_feat, fake_feat in zip(real_feats, fake_feats):
                feat_loss += F.l1_loss(fake_feat, real_feat)
        adv_loss /= len(fake_preds)
        feat_loss /= sum(len(f) for f in real_featss)
        assert isinstance(adv_loss, Tensor)
        assert isinstance(feat_loss, Tensor)
        recon_loss = self.reconstruction_loss(x, y)["loss"]
        loss = (
            self.adversarial_loss_weight * adv_loss
            + self.feature_loss_weight * feat_loss
            + self.reconstruction_loss_weight * recon_loss
            + self.codebook_loss_weight * codebook_loss
            + self.commitment_loss_weight * commit_loss
        )
        return {
            "loss": loss,
            "adv_loss": adv_loss,
            "feat_loss": feat_loss,
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commit_loss": commit_loss,
        }

    @override
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        optimizers = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise ValueError("NACLightningModule.configure_optimizers() must return two optimizers.")
        generator_optimizer, discriminator_optimizer = optimizers

        noisy, clean, _ = batch
        x = torch.cat([clean, noisy], dim=0)
        y, _, codebook_loss, commit_loss = self.generator(x)

        if not codebook_loss.isfinite() or codebook_loss > 1e6:
            raise ValueError(f"Codebook loss exploded: {codebook_loss.item()}")

        disc_loss = self.discriminator_step(x, y)
        if stage == "train":
            discriminator_optimizer.zero_grad()
            self.manual_backward(disc_loss)
            self.clip_gradients(discriminator_optimizer, self.discriminator_grad_clip)  # type: ignore
            discriminator_optimizer.step()

        gen_losses = self.generator_step(x, y, codebook_loss, commit_loss)
        if stage == "train":
            generator_optimizer.zero_grad()
            self.manual_backward(gen_losses["loss"])
            self.clip_gradients(generator_optimizer, self.generator_grad_clip)  # type: ignore
            generator_optimizer.step()

        all_losses = {"disc_loss": disc_loss, **gen_losses}
        metric_vals = compute_metrics(y, x, metrics)
        clean_out = y[: clean.shape[0]]
        debug_samples = {"input": clean, "output": clean_out} if self.current_epoch == 0 else {"output": clean_out}
        return all_losses, metric_vals, debug_samples

    def configure_optimizers(self) -> tuple[Optimizer, Optimizer]:
        """Configure optimizers.

        Returns:
            Tuple of generator and discriminator optimizers.
        """
        generator_optimizer = self.generator_optimizer(self.generator.parameters())
        discriminator_optimizer = self.discriminator_optimizer(self.discriminator.parameters())
        return generator_optimizer, discriminator_optimizer

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the generator."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        # pad to multiple of model.downsampling_factor
        n_pad = (self.generator.downsampling_factor - x.shape[-1]) % self.generator.downsampling_factor
        x_pad = F.pad(x, (0, n_pad))
        y_pad, _, _, _ = self.generator(x_pad)
        return y_pad[..., : y_pad.shape[-1] - n_pad]


class ADDSELightningModule(BaseLightningModule, ConfigureOptimizersMixin):
    """ADDSE Lightning module."""

    def __init__(
        self,
        nac_cfg: str,
        nac_ckpt: str,
        model: ADDSERQDiT,
        num_steps: int,
        block_size: int,
        optimizer: Callable[[Iterator[nn.Parameter]], Optimizer] = Adam,
        lr_scheduler: Mapping[str, Any] | None = None,
        val_metrics: Mapping[str, BaseMetric] | None = None,
        test_metrics: Mapping[str, BaseMetric] | None = None,
        log_cfg: LogConfig | None = None,
        debug_sample: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the ADDSE Lightning module."""
        super().__init__()
        self.nac, self.mask_token = load_nac(nac_cfg, nac_ckpt)
        self.model = model
        self.num_steps = num_steps
        self.block_size = block_size
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = debug_sample

    @override
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        x, y, _ = batch
        if stage == "test":
            # pad to multiple of nac.downsampling_factor
            n_pad = (self.nac.downsampling_factor - x.shape[-1]) % self.nac.downsampling_factor
            x, y = F.pad(x, (0, n_pad)), F.pad(y, (0, n_pad))
        xy_tok, xy_q = self.nac.encode(torch.cat([x, y]), no_sum=True, domain="q")
        x_tok, y_tok = xy_tok.chunk(2)
        x_q, y_q = xy_q.chunk(2)
        loss = {"loss": self.loss(x_q, y_q, y_tok)}
        if metrics or stage == "val" and self.debug_sample is not None and batch_idx == self.debug_sample[0]:
            y_hat_tok = self.solve(x_tok, x_q, self.num_steps)
            assert isinstance(y_hat_tok, Tensor)
            y_hat = self.nac.decode(y_hat_tok, domain="code")
            y_decoded = self.nac.decode(y_q, no_sum=True, domain="q")
            metric_vals = compute_metrics(y_hat, y, metrics)
            debug_samples = (
                {"input": x, "reference": y, "output": y_hat, "reference_decoded": y_decoded}
                if self.current_epoch == 0
                else {"output": y_hat}
            )
            return loss, metric_vals, debug_samples
        return loss, {}, {}

    def loss(self, x_q: Tensor, y_q: Tensor, y_tok: Tensor) -> Tensor:
        r"""Compute the $\lambda$-denoising cross-entropy loss.

        Args:
            x_q: Noisy speech embeddings. Shape `(batch_size, emb_channels, num_codebooks, seq_len)`.
            y_q: Clean speech embeddings. Shape `(batch_size, emb_channels, num_codebooks, seq_len)`.
            y_tok: Clean speech tokens. Shape `(batch_size, num_codebooks, seq_len)`.

        Returns:
            The $\lambda$-denoising cross-entropy loss.
        """
        lambd = torch.rand(y_tok.shape[0], device=y_tok.device)  # (B,)
        mask = torch.rand(y_tok.shape, device=y_tok.device) < lambd[:, None, None]  # (B, K, L)
        y_lambda_q = y_q.masked_fill(mask[:, None], 0)  # (B, C, K, L)
        log_score = self.log_score(y_lambda_q, x_q)  # (B, K, L, V)
        loss = torch.zeros(y_tok.shape, device=y_tok.device, dtype=log_score.dtype)  # (B, K, L)
        loss[mask] = torch.gather(log_score[mask], -1, y_tok[mask][:, None]).squeeze(-1)  # (N,)
        loss = -loss.mean(dim=(-1, -2)) / lambd  # (B,)
        return loss.mean()

    @torch.no_grad()
    def solve(
        self, x_tok: Tensor, x_q: Tensor, num_steps: int, return_nfe: bool = False
    ) -> Tensor | tuple[Tensor, int]:
        """Sample assuming a log-linear noise schedule and an absorbing transition matrix."""
        assert not self.model.training, "Model must be in eval mode to sample."
        y_tok = torch.full_like(x_tok, self.mask_token)  # (B, K, L)
        y_q = torch.zeros_like(x_q)  # (B, C, K, L)
        update_rate = 1 / num_steps / torch.linspace(1.0, 0.0, num_steps + 1, device=x_tok.device)
        changed = torch.ones(x_tok.shape[0], dtype=torch.bool, device=x_tok.device)  # (B,)
        score = torch.zeros(*x_tok.shape, self.mask_token + 1, device=x_tok.device)  # (B, K, L, V + 1)
        nfe = 0
        for i in range(num_steps):
            if changed.any():
                mask = y_tok == self.mask_token  # (B, K, L)
                # set mask tokens to zero when decoding to avoid codebook lookup error
                y_q[changed] = self.nac.quantizer.decode(
                    y_tok[changed].masked_fill(mask[changed], 0), output_no_sum=True, domain="code"
                )
                # then correct the decoded embeddings by setting them to zero where masked
                y_q[changed] = y_q[changed].masked_fill(mask[changed, None], 0)  # (B, C, K, L)
                score[changed, ..., :-1] = self.log_score(y_q[changed], x_q[changed]).exp()  # (B, K, L, V)
                score_mask = score[mask]  # (N, V)
                nfe += 1
            probs_mask = score_mask * update_rate[i]  # (N, V)
            probs_mask[:, -1] = 1 - update_rate[i]  # (N,)
            y_tok_old = y_tok.clone()  # (B, K, L)
            y_tok[mask] = torch.multinomial(probs_mask, 1).squeeze(-1)  # (N,)
            changed = (y_tok != y_tok_old).any(dim=(-1, -2))  # (B,)
        if return_nfe:
            return y_tok, nfe
        return y_tok

    def log_score(self, y_q: Tensor, x_q: Tensor) -> Tensor:
        """Estimate the score function."""
        score = process_in_blocks((y_q, x_q), self.block_size, self.model)  # (B, V, K, L)
        return score.moveaxis(1, -1).log_softmax(dim=-1)  # (B, K, L, V)

    def forward(self, x: Tensor, return_nfe: bool = False) -> Tensor | tuple[Tensor, int]:
        """Enhance the input audio."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        # pad to multiple of nac.downsampling_factor
        n_pad = (self.nac.downsampling_factor - x.shape[-1]) % self.nac.downsampling_factor
        x_pad = F.pad(x, (0, n_pad))
        x_tok, x_q = self.nac.encode(x_pad, no_sum=True, domain="q")
        output = self.solve(x_tok, x_q, self.num_steps, return_nfe=return_nfe)
        if return_nfe:
            assert isinstance(output, tuple)
            y_tok, nfe = output
        else:
            assert isinstance(output, Tensor)
            y_tok = output
        y_pad = self.nac.decode(y_tok, domain="code")
        if return_nfe:
            return y_pad[..., : y_pad.shape[-1] - n_pad], nfe
        return y_pad[..., : y_pad.shape[-1] - n_pad]


class NACSELightningModule(BaseLightningModule, ConfigureOptimizersMixin):
    """Lightning module for speech enhancement using NAC-domain direct prediction."""

    def __init__(
        self,
        nac_cfg: str,
        nac_ckpt: str,
        nac_domain: str,
        nac_no_sum: bool,
        model: nn.Module,
        block_size: int,
        optimizer: Callable[[Iterator[nn.Parameter]], Optimizer] = Adam,
        lr_scheduler: Mapping[str, Any] | None = None,
        val_metrics: Mapping[str, BaseMetric] | None = None,
        test_metrics: Mapping[str, BaseMetric] | None = None,
        log_cfg: LogConfig | None = None,
        debug_sample: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the NAC-domain Lightning module."""
        super().__init__()
        self.nac, _ = load_nac(nac_cfg, nac_ckpt)
        self.nac_domain = nac_domain
        self.nac_no_sum = nac_no_sum
        self.model = model
        self.block_size = block_size
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = debug_sample

    @override
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        x, y, _ = batch
        _, xy_q = self.nac.encode(torch.cat([x, y]), no_sum=self.nac_no_sum, domain=self.nac_domain)
        x_q, y_q = xy_q.chunk(2)
        y_hat_q = process_in_blocks((x_q,), self.block_size, self.model)
        loss = {"loss": (y_hat_q - y_q).pow(2).mean()}
        if metrics or stage == "val" and self.debug_sample is not None and batch_idx == self.debug_sample[0]:
            y_hat = self.nac.decode(y_hat_q, no_sum=self.nac_no_sum, domain=self.nac_domain)
            y_decoded = self.nac.decode(y_q, no_sum=self.nac_no_sum, domain=self.nac_domain)
            metric_vals = compute_metrics(y_hat, y, metrics)
            debug_samples = (
                {"input": x, "reference": y, "output": y_hat, "reference_decoded": y_decoded}
                if self.current_epoch == 0
                else {"output": y_hat}
            )
            return loss, metric_vals, debug_samples
        return loss, {}, {}

    def forward(self, x: Tensor) -> Tensor:
        """Enhance the input audio."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        # pad to multiple of nac.downsampling_factor
        n_pad = (self.nac.downsampling_factor - x.shape[-1]) % self.nac.downsampling_factor
        x_pad = F.pad(x, (0, n_pad))
        _, x_q = self.nac.encode(x_pad, no_sum=self.nac_no_sum, domain=self.nac_domain)
        y_pad = process_in_blocks((x_q,), self.block_size, self.model)
        y_pad = self.nac.decode(y_pad, no_sum=self.nac_no_sum, domain=self.nac_domain)
        return y_pad[..., : y_pad.shape[-1] - n_pad]


class EDMNACSELightningModule(BaseLightningModule, ConfigureOptimizersMixin, EDMMixin):
    """Lightning module for speech enhancement using NAC-domain EDM-style diffusion."""

    def __init__(
        self,
        nac_cfg: str,
        nac_ckpt: str,
        nac_domain: str,
        nac_no_sum: bool,
        nac_stack: bool,
        model: ADDSERQDiT,
        num_steps: int,
        block_size: int,
        norm_factor: float = 2.3,
        sigma_data: float = 0.5,
        p_mean: float = 0.0,
        p_sigma: float = 1.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        s_noise: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        optimizer: Callable[[Iterator[nn.Parameter]], Optimizer] = Adam,
        lr_scheduler: Mapping[str, Any] | None = None,
        val_metrics: Mapping[str, BaseMetric] | None = None,
        test_metrics: Mapping[str, BaseMetric] | None = None,
        log_cfg: LogConfig | None = None,
        debug_sample: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the NAC-domain EDM-style Lightning module."""
        super().__init__()
        self.nac, _ = load_nac(nac_cfg, nac_ckpt)
        self.nac_domain = nac_domain
        self.nac_no_sum = nac_no_sum
        self.nac_stack = nac_stack
        self.model = model
        self.num_steps = num_steps
        self.block_size = block_size
        self.norm_factor = norm_factor
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_sigma = p_sigma
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.s_noise = s_noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = debug_sample

    @override
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        x, y, _ = batch
        _, xy_q = self.nac.encode(torch.cat([x, y]), no_sum=self.nac_no_sum, domain=self.nac_domain)
        x_q, y_q = xy_q.chunk(2)
        x_q_norm = x_q * self.sigma_data / self.norm_factor  # See section B.1 "Latent diffusion" in EDM2 paper
        y_q_norm = y_q * self.sigma_data / self.norm_factor  # See section B.1 "Latent diffusion" in EDM2 paper
        x_q_norm = x_q_norm.flatten(1, 2).unsqueeze(2) if self.nac_stack else x_q_norm
        y_q_norm = y_q_norm.flatten(1, 2).unsqueeze(2) if self.nac_stack else y_q_norm
        loss = {"loss": self.loss(x_q_norm, y_q_norm)}
        if metrics or stage == "val" and self.debug_sample is not None and batch_idx == self.debug_sample[0]:
            y_hat_q = self.solve(x_q_norm, self.num_steps)
            y_hat_q = y_hat_q.squeeze(2).unflatten(1, (y_q.shape[1], y_q.shape[2])) if self.nac_stack else y_hat_q
            y_hat_q = y_hat_q * self.norm_factor / self.sigma_data
            y_hat = self.nac.decode(y_hat_q, no_sum=self.nac_no_sum, domain=self.nac_domain)
            y_decoded = self.nac.decode(y_q, no_sum=self.nac_no_sum, domain=self.nac_domain)
            metric_vals = compute_metrics(y_hat, y, metrics)
            debug_samples = (
                {"input": x, "reference": y, "output": y_hat, "reference_decoded": y_decoded}
                if self.current_epoch == 0
                else {"output": y_hat}
            )
            return loss, metric_vals, debug_samples
        return loss, {}, {}

    @override
    def denoiser(self, y: Tensor, x: Tensor, sigma: Tensor) -> Tensor:
        return process_in_blocks((y, x), self.block_size, functools.partial(super().denoiser, sigma=sigma))

    def forward(self, x: Tensor, num_steps: int | None = None) -> Tensor:
        """Enhance the input audio."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        # pad to multiple of nac.downsampling_factor
        n_pad = (self.nac.downsampling_factor - x.shape[-1]) % self.nac.downsampling_factor
        x_pad = F.pad(x, (0, n_pad))
        _, x_q = self.nac.encode(x_pad, no_sum=self.nac_no_sum, domain=self.nac_domain)
        x_q_norm = x_q * self.sigma_data / self.norm_factor  # See section B.1 "Latent diffusion" in EDM2 paper
        x_q_norm = x_q_norm.flatten(1, 2).unsqueeze(2) if self.nac_stack else x_q_norm
        y_q_norm = self.solve(x_q_norm, self.num_steps if num_steps is None else num_steps)
        y_q = y_q_norm.squeeze(2).unflatten(1, (x_q.shape[1], x_q.shape[2])) if self.nac_stack else y_q_norm
        y_q = y_q * self.norm_factor / self.sigma_data
        y_pad = self.nac.decode(y_q, no_sum=self.nac_no_sum, domain=self.nac_domain)
        return y_pad[..., : y_pad.shape[-1] - n_pad]


class EDMSELightningModule(BaseLightningModule, ConfigureOptimizersMixin, EDMMixin):
    """Lightning module for speech enhancement using STFT-domain EDM-style diffusion."""

    def __init__(
        self,
        model: ADM,
        stft: STFT,
        num_steps: int = 30,
        sigma_data: float = 0.5,
        p_mean: float = 0.0,
        p_sigma: float = 1.0,
        s_churn: float = 0.0,
        s_min: float = 0.0,
        s_max: float = float("inf"),
        s_noise: float = 1.0,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        optimizer: Callable[[Iterator[nn.Parameter]], Optimizer] = Adam,
        lr_scheduler: Mapping[str, Any] | None = None,
        val_metrics: Mapping[str, BaseMetric] | None = None,
        test_metrics: Mapping[str, BaseMetric] | None = None,
        log_cfg: LogConfig | None = None,
        debug_sample: tuple[int, int] | None = None,
    ) -> None:
        """Initialize the NAC-domain EDM-style Lightning module."""
        super().__init__()
        self.model: ADM = model
        self.stft = stft
        self.num_steps = num_steps
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_sigma = p_sigma
        self.s_churn = s_churn
        self.s_min = s_min
        self.s_max = s_max
        self.s_noise = s_noise
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg
        self.debug_sample = debug_sample

    @override
    def step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
        metrics: Mapping[str, BaseMetric] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
        x, y, _ = batch
        x_stft, y_stft = self.transform(x), self.transform(y)
        # set std to sigma_data
        factor = x_stft.std(dim=(1, 2, 3), keepdim=True) / self.sigma_data
        x_stft, y_stft = x_stft / factor, y_stft / factor
        loss = {"loss": self.loss(x_stft, y_stft)}
        if metrics or stage == "val" and self.debug_sample is not None and batch_idx == self.debug_sample[0]:
            y_hat_stft = self.solve(x_stft, self.num_steps)
            # undo std scaling
            y_hat_stft = y_hat_stft * factor
            y_hat = self.inverse_transform(y_hat_stft, n=x.shape[-1])
            metric_vals = compute_metrics(y_hat, y, metrics)
            debug = {"input": x, "reference": y, "output": y_hat} if self.current_epoch == 0 else {"output": y_hat}
            return loss, metric_vals, debug
        return loss, {}, {}

    def transform(self, x: Tensor) -> Tensor:
        """Compute the STFT and compress."""
        x = self.stft(x)
        return torch.polar(x.abs().sqrt(), x.angle())

    def inverse_transform(self, x: Tensor, n: int) -> Tensor:
        """Decompress and compute the inverse STFT."""
        x = torch.polar(x.abs().square(), x.angle())
        return self.stft.inverse(x, n=n)

    def forward(self, x: Tensor, num_steps: int | None = None) -> Tensor:
        """Enhance the input audio."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        x_stft = self.transform(x)
        # set std to sigma_data
        factor = x_stft.std(dim=(1, 2, 3), keepdim=True) / self.sigma_data
        x_stft = x_stft / factor
        # pad to multiple of model.downsampling_factor
        n_pad = (self.model.downsampling_factor - x_stft.shape[-1]) % self.model.downsampling_factor
        x_stft_pad = F.pad(x_stft, (0, n_pad))
        y_stft_pad = self.solve(x_stft_pad, self.num_steps if num_steps is None else num_steps)
        y_stft = y_stft_pad[..., : y_stft_pad.shape[-1] - n_pad]
        # undo std scaling
        return self.inverse_transform(y_stft * factor, n=x.shape[-1])


class DataModule(L.LightningDataModule):
    """Data module."""

    def __init__(
        self,
        train_dataset: Callable[[], Dataset],
        train_dataloader: Callable[[Dataset], DataLoader],
        val_dataset: Callable[[], Dataset] | None = None,
        val_dataloader: Callable[[Dataset], DataLoader] | None = None,
        test_dataset: Callable[[], Dataset] | None = None,
        test_dataloader: Callable[[Dataset], DataLoader] | None = None,
    ) -> None:
        """Initialize the data module.

        Args:
            train_dataset: Function to initialize the training dataset.
            val_dataset: Function to initialize the validation dataset.
            test_dataset: Function to initialize the test dataset.
            train_dataloader: Function to initialize the training dataloader.
            val_dataloader: Function to initialize the validation dataloader.
            test_dataloader: Function to initialize the test dataloader.
        """
        super().__init__()
        self.train_dataset_fn = train_dataset
        self.val_dataset_fn = val_dataset
        self.test_dataset_fn = test_dataset
        self.train_dataloader_fn = train_dataloader
        self.val_dataloader_fn = val_dataloader
        self.test_dataloader_fn = test_dataloader
        self.train_dset: Dataset | None = None
        self.val_dset: Dataset | None = None
        self.test_dset: Dataset | None = None
        self._state_dict: dict[str, Any] | None = None

    def setup(self, stage: str) -> None:
        """Setup the data module.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        self.train_dset = self.train_dataset_fn()
        self.val_dset = None if self.val_dataset_fn is None else self.val_dataset_fn()
        self.test_dset = None if self.test_dataset_fn is None else self.test_dataset_fn()
        if self.trainer is not None and (
            isinstance(self.train_dset, DynamicMixingDataset)
            and not isinstance(self.trainer.limit_train_batches, int)
            or isinstance(self.val_dset, DynamicMixingDataset)
            and not isinstance(self.trainer.limit_val_batches, int)
            or isinstance(self.test_dset, DynamicMixingDataset)
            and not isinstance(self.trainer.limit_test_batches, int)
        ):
            raise ValueError(
                "DynamicMixingDataset requires a fixed number of batches. Set `limit_<stage>_batches` to an integer."
            )

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns:
            The training dataloader.
        """
        assert self.train_dset is not None
        train_dataloader = self.train_dataloader_fn(self.train_dset)
        if self._state_dict is not None and isinstance(train_dataloader, AudioStreamingDataLoader):
            train_dataloader.load_state_dict(self._state_dict)
            self._state_dict = None
        return train_dataloader

    def val_dataloader(self) -> DataLoader | list:
        """Get the validation dataloader.

        Returns:
            The validation dataloader or an empty list if no validation dataset was provided at initialization.
        """
        if self.val_dataloader_fn is None or self.val_dset is None:
            return []
        return self.val_dataloader_fn(self.val_dset)

    def test_dataloader(self) -> DataLoader | list:
        """Get the test dataloader.

        Returns:
            The test dataloader or an empty list if no test dataset was provided at initialization.
        """
        if self.test_dataloader_fn is None or self.test_dset is None:
            return []
        return self.test_dataloader_fn(self.test_dset)

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict of the data module."""
        if self.trainer is not None and isinstance(self.trainer.train_dataloader, AudioStreamingDataLoader):
            return self.trainer.train_dataloader.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict of the data module."""
        self._state_dict = state_dict


def compute_metrics(x: Tensor, y: Tensor, metrics: Mapping[str, BaseMetric] | None = None) -> dict[str, float]:
    """Compute validation or test metrics.

    Args:
        x: Signal to evaluate. Shape `(batch_size, num_channels, num_samples)`.
        y: Reference signal for the metrics. Shape `(batch_size, num_channels, num_samples)`.
        metrics: Metrics to compute.

    Returns:
        Dictionary with computed metrics.
    """
    if not (x.ndim == y.ndim == 3) or x.shape != y.shape:
        raise ValueError(f"Inputs must be 3-dimensional and have the same shape. Got {x.shape} and {y.shape}.")
    return {
        metric_name: sum(metric(x_i, y_i) for x_i, y_i in zip(x, y)) / y.shape[0]
        for metric_name, metric in (metrics or {}).items()
    }


def load_nac(cfg_path: str, ckpt_path: str) -> tuple[NAC, int]:
    """Load a pretrained neural audio codec."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    nac: NAC = instantiate(cfg["lm"]["generator"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {k.removeprefix("generator."): v for k, v in ckpt["state_dict"].items() if k.startswith("generator.")}
    nac.load_state_dict(state_dict)
    nac.eval()
    for param in nac.parameters():
        param.requires_grad = False
    codebook_size: int = nac.quantizer.codebooks[0].codebook.weight.shape[0]  # type: ignore
    return nac, codebook_size


def process_in_blocks(args: tuple[Tensor, ...], block_size: int, fn: Callable[..., Tensor]) -> Tensor:
    """Process the inputs in blocks."""
    assert all(arg.shape[-1] == args[0].shape[-1] for arg in args), "All inputs must have same size along last dim."
    blocks = [fn(*(arg[..., i : i + block_size] for arg in args)) for i in range(0, args[0].shape[-1], block_size)]
    return torch.cat(blocks, dim=-1)
