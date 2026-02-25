from abc import abstractmethod
from collections.abc import Collection
from typing import Literal, override

import torch
import torch.nn as nn

from .stft import STFT
from .utils import flatten_dict, mel_filters


class BaseLoss(nn.Module):
    """Base class for losses."""

    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the loss.

        This method should not be called directly. Use `forward` instead.
        """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute the loss.

        Validates inputs and calls `compute`.

        Args:
            x: Predicted signal. Shape `(batch_size, num_channels, num_samples)`.
            y: Target signal. Shape `(batch_size, num_channels, num_samples)`.

        Returns:
            Loss dictionary.
        """
        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError(f"Inputs must be torch tensors. Got {type(x).__name__} and {type(y).__name__}.")
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError(f"Inputs must be 3-dimensional. Got shapes {x.shape} and {y.shape}.")
        if x.shape != y.shape:
            raise ValueError(f"Inputs must have the same shape. Got {x.shape} and {y.shape}.")
        output = self.compute(x, y)
        if isinstance(output, torch.Tensor):
            output = {"loss": output}
        elif not isinstance(output, dict):
            raise TypeError(f"Loss output must be a torch tensor or a dictionary. Got {type(output).__name__}.")
        if "loss" not in output:
            raise ValueError("The output loss dictionary must contain the key 'loss'.")
        for key, value in output.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    "All values in the output loss dictionary must be torch tensors. "
                    f"Got {type(value).__name__} for key '{key}'."
                )
            if value.ndim != 0:
                raise ValueError(
                    "All values in the output loss dictionary must be scalars. "
                    f"Got shape {value.shape} for key '{key}'."
                )
        return output


class MultiTaskLoss(BaseLoss):
    """Multi-task loss."""

    def __init__(
        self,
        losses: Collection[BaseLoss],
        weights: Collection[float] | None = None,
        names: Collection[str] | None = None,
    ) -> None:
        """Initialize the multi-task loss."""
        super().__init__()
        assert losses, "`losses` must not be empty."
        assert weights is None or len(weights) == len(losses), "`weights` must have the same length as `losses`."
        assert names is None or len(names) == len(losses), "`names` must have the same length as `losses`."
        self.losses = nn.ModuleList(losses)
        self.weights = [1.0 for _ in losses] if weights is None else weights
        self.names = [f"loss_{i}" for i, _ in enumerate(losses)] if names is None else names

    @override
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        output = {}
        total = 0.0
        for loss, weight, name in zip(self.losses, self.weights, self.names):
            output[name] = loss(x, y)
            total = total + weight * output[name]["loss"]
        output["loss"] = total
        return flatten_dict(output)


class SDRLoss(BaseLoss):
    """Signal-to-distortion ratio (SDR) loss."""

    def __init__(self, scale_invariant: bool = False, zero_mean: bool = False, eps: float = 1e-7) -> None:
        """Initialize the SDR loss.

        Args:
            scale_invariant: If `True`, computes the scale-invariant signal-to-distortion ratio (SI-SDR).
            zero_mean: If `True`, subtracts the mean from the inputs before computing the loss.
            eps: Small value for numerical stability.
        """
        super().__init__()
        self.scale_invariant = scale_invariant
        self.zero_mean = zero_mean
        self.eps = eps

    @override
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.zero_mean:
            x = x - x.mean(dim=(1, 2), keepdim=True)
            y = y - y.mean(dim=(1, 2), keepdim=True)
        if self.scale_invariant:
            y = y * (x * y).sum((1, 2), keepdim=True) / (y**2).sum((1, 2), keepdim=True).clamp(min=self.eps)
        num = (y**2).sum((1, 2), keepdim=True).clamp(min=self.eps)
        den = ((x - y) ** 2).sum((1, 2), keepdim=True).clamp(min=self.eps)
        return -10 * torch.log10(num / den).mean()


class MelSpecLoss(BaseLoss):
    """Mel-spectrogram loss."""

    def __init__(
        self,
        n_mels: int = 64,
        frame_length: int = 512,
        hop_length: int | None = None,
        n_fft: int | None = None,
        window: str = "flattop",
        fs: int = 16000,
        compression: float = 2.0,
        log: bool = True,
        power: float = 1.0,
        eps: float = 1e-7,
        mel_norm: Literal["slaney", "consistent"] | None = "consistent",
        stft_norm: bool = True,
    ) -> None:
        """Initialize the mel-spectrogram loss."""
        super().__init__()
        n_fft = frame_length if n_fft is None else n_fft
        self.stft = STFT(frame_length=frame_length, hop_length=hop_length, n_fft=n_fft, window=window, norm=stft_norm)
        self.compression = compression
        self.log = log
        self.power = power
        self.eps = eps
        filters, _ = mel_filters(n_filters=n_mels, n_fft=n_fft, fs=fs, norm=mel_norm)
        self.register_buffer("filters", filters, persistent=False)

    @override
    def compute(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_stft, y_stft = self.stft(x), self.stft(y)
        x_mag, y_mag = x_stft.abs().pow(self.compression), y_stft.abs().pow(self.compression)
        x_mel = torch.einsum("ij,bcjk->bcik", self.filters, x_mag)
        y_mel = torch.einsum("ij,bcjk->bcik", self.filters, y_mag)
        if self.log:
            x_mel, y_mel = x_mel.clamp(min=self.eps).log(), y_mel.clamp(min=self.eps).log()
        loss = (x_mel - y_mel).abs()
        if self.power != 1.0:
            loss = loss.pow(self.power)
        return loss.mean()


class MSMelSpecLoss(MultiTaskLoss):
    """Multi-scale mel-spectrogram loss."""

    def __init__(
        self,
        n_mels: int | Collection[int] = (4, 8, 16, 32, 64, 128, 256),
        frame_lengths: Collection[int] = (31, 67, 127, 257, 509, 1021, 2053),
        hop_lengths: Collection[int | None] | None = None,
        n_ffts: Collection[int | None] | None = None,
        weights: Collection[float] | None = None,
        window: str = "flattop",
        fs: int = 16000,
        compression: float = 2.0,
        log: bool = True,
        power: float = 1.0,
        eps: float = 1e-7,
        mel_norm: Literal["slaney", "consistent"] | None = "consistent",
        stft_norm: bool = True,
    ) -> None:
        """Initialize the multi-scale mel-spectrogram loss."""
        n_mels = [n_mels] * len(frame_lengths) if isinstance(n_mels, int) else n_mels
        hop_lengths = [None] * len(frame_lengths) if hop_lengths is None else hop_lengths
        n_ffts = [None] * len(frame_lengths) if n_ffts is None else n_ffts
        weights = [1.0 / len(frame_lengths)] * len(frame_lengths) if weights is None else weights
        if len(n_mels) != len(frame_lengths):
            raise ValueError("`n_mels` must have the same length as `frame_lengths`.")
        if len(hop_lengths) != len(frame_lengths):
            raise ValueError("`hop_lengths` must have the same length as `frame_lengths`.")
        if len(n_ffts) != len(frame_lengths):
            raise ValueError("`n_ffts` must have the same length as `frame_lengths`.")
        if len(weights) != len(frame_lengths):
            raise ValueError("`weights` must have the same length as `frame_lengths`.")
        losses = [
            MelSpecLoss(
                n_mels=n_mel,
                frame_length=frame_length,
                hop_length=hop_length,
                n_fft=n_fft,
                window=window,
                fs=fs,
                compression=compression,
                log=log,
                power=power,
                eps=eps,
                mel_norm=mel_norm,
                stft_norm=stft_norm,
            )
            for n_mel, frame_length, hop_length, n_fft in zip(n_mels, frame_lengths, hop_lengths, n_ffts)
        ]
        names = [f"melspec_{frame_length}" for frame_length in frame_lengths]
        super().__init__(losses, weights=weights, names=names)
