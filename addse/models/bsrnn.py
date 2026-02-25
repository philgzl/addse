import functools
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import BandMerge, BandSplit, LayerNorm
from ..stft import STFT
from ..utils import build_subbands


class BSRNN(nn.Module):
    """Band-split RNN (BSRNN) [^1] [^2] [^3].

    [^1]: Y. Luo and J. Yu, "Music source separation with band-split RNN", IEEE/ACM TASLP, 2023.
    [^2]:
        J. Yu and Y. Luo, "Efficient monaural speech enhancement with universal sample rate band-split RNN", IEEE
        ICASSP, 2023.
    [^3]:
        J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, "High fidelity speech enhancement with band-split RNN",
        INTERSPEECH, 2023.
    """

    def __init__(
        self,
        stft: STFT | None = None,
        fs: int = 16000,
        input_channels: int = 1,
        output_channels: int = 1,
        num_channels: int = 32,
        num_layers: int = 6,
        causal: bool = False,
        subbands: Iterable[tuple[float, int]] = [(100.0, 10), (200.0, 10), (500.0, 6), (1000.0, 2)],
        residual: bool = False,
        norm: Callable[[int], nn.Module] | None = None,
    ) -> None:
        """Initialize BSRNN.

        Args:
            stft: STFT module.
            fs: Sampling rate.
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            num_channels: Number of internal channels. Denoted as _N_ in the paper.
            num_layers: Number of dual-path modelling layers.
            causal: Whether to use unidirectional RNNs along the time axis.
            subbands: List of tuples `(bandwidth, number)`, where `bandwidth` is the bandwidth of the subband in Hz
                and `number` is the number of subbands with that bandwidth.
            residual: Whether to predict a residual STFT in addition to the mask. The residual STFT is added after
                applying the mask to the input STFT.
            norm: Normalization module to use throughout the network. If `None`, defaults to
                [LayerNorm][addse.layers.LayerNorm] with `causal=causal`. If a non-causal normalization module is
                provided, the network is not causal, even if `causal=True`.
        """
        super().__init__()
        self.stft = STFT() if stft is None else stft
        subband_idx = build_subbands(self.stft.n_fft, fs, subbands)
        norm = functools.partial(LayerNorm, causal=causal) if norm is None else norm
        self.band_split = BandSplit(subband_idx, input_channels, num_channels, norm)
        self.time_blocks = nn.ModuleList(
            [BSRNNRNNBlock(num_channels, 2 * num_channels, causal, -1, norm) for _ in range(num_layers)]
        )
        self.freq_blocks = nn.ModuleList(
            [BSRNNRNNBlock(num_channels, 2 * num_channels, causal, -2, norm) for _ in range(num_layers)]
        )
        self.band_merge = BandMerge(
            subband_idx, input_channels, output_channels, num_channels, norm, BSRNNMLP, residual
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Shape `(batch_size, input_channels, num_samples)`.

        Returns:
            Enhanced tensor. Shape `(batch_size, output_channels, num_samples)`.
        """
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        stft = self.stft(x)  # (B, C_in, F, T)
        h = self.band_split(stft)  # (B, N, K, T)
        for time_block, freq_block in zip(self.time_blocks, self.freq_blocks):
            h = h + time_block(h)  # (B, N, K, T)
            h = h + freq_block(h)  # (B, N, K, T)
        mask, residual = self.band_merge(h)  # (B, C_in, C_out, F', T), (B, C_out, F', T)
        # pad if subbands do not span full frequency range
        mask = F.pad(mask, (0, 0, 0, stft.shape[-2] - mask.shape[-2]), value=1.0)  # (B, C_in, C_out, F, T)
        stft = torch.einsum("bift,bijft->bjft", stft, mask)
        if residual is not None:
            # pad if subbands do not span full frequency range
            residual = F.pad(residual, (0, 0, 0, stft.shape[-2] - residual.shape[-2]), value=0.0)  # (B, C_out, F, T)
            stft = stft + residual
        return self.stft.inverse(stft, n=x.shape[-1])


class BSRNNRNNBlock(nn.Module):
    """RNN block used in BSRNN."""

    def __init__(
        self, num_channels: int, hidden_channels: int, causal: bool, seq_dim: int, norm: Callable[[int], nn.Module]
    ) -> None:
        """Initialize the BSRNN RNN block."""
        super().__init__()
        if seq_dim not in (-1, -2, 2, 3):
            raise ValueError(f"`seq_dim` must be -1, -2, 2, or 3. Got {seq_dim}.")
        bidirectional = not causal if seq_dim in (-1, 3) else True
        self.norm = norm(num_channels)
        self.lstm = nn.LSTM(num_channels, hidden_channels, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(2 * hidden_channels if bidirectional else hidden_channels, num_channels)
        self.seq_dim = seq_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B, N, _, _ = x.shape
        x = self.norm(x)  # (B, N, K, T)
        x = x.moveaxis(self.seq_dim, -1).moveaxis(1, -1)  # (B, K/T, T/K, N)
        x = x.reshape(-1, x.shape[2], N)  # (B * K/T, T/K, N)
        x, _ = self.lstm(x)  # (B * K/T, T/K, hidden_channels)
        x = self.fc(x)  # (B * K/T, T/K, N)
        x = x.reshape(B, -1, x.shape[1], N)  # (B, K/T, T/K, N)
        return x.moveaxis(-1, 1).moveaxis(-1, self.seq_dim)  # (B, N, K, T)


class BSRNNMLP(nn.Module):
    """Multi-Layer perceptron (MLP) used in BSRNN."""

    def __init__(self, input_channels: int, output_channels: int, norm: Callable[[int], nn.Module]) -> None:
        """Initialize the BSRNN MLP."""
        super().__init__()
        self.norm = norm(input_channels)
        self.fc_1 = nn.Conv1d(input_channels, 4 * input_channels, 1)
        self.act_1 = nn.Tanh()
        self.fc_2 = nn.Conv1d(4 * input_channels, 2 * output_channels, 1)
        self.act_2 = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.norm(x)
        x = self.fc_1(x)
        x = self.act_1(x)
        x = self.fc_2(x)
        return self.act_2(x)
