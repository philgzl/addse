import functools
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import LayerNorm


class ConvTasNet(nn.Module):
    """Conv-TasNet [^1].

    Consists of an encoder, a temporal convolutional network (TCN), and a decoder.

    [^1]:
        Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation",
        in IEEE/ACM TASLP, 2019.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_filters: int = 512,
        filter_size: int = 32,
        hop_size: int | None = None,
        bottleneck_channels: int = 128,
        hidden_channels: int = 512,
        skip_channels: int = 128,
        kernel_size: int = 3,
        layers: int = 8,
        repeats: int = 3,
        causal: bool = False,
        norm: Callable[[int], nn.Module] | None = None,
    ) -> None:
        """Initialize Conv-TasNet.

        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            num_filters: Number of filters in the encoder. Denoted as _N_ in the paper.
            filter_size: Encoder filter length. Denoted as _L_ in the paper.
            hop_size: Encoder hop size. If `None`, defaults to `encoder_kernel_size // 2`.
            bottleneck_channels: Number of bottleneck channels in the TCN. Denoted as _B_ in the paper.
            hidden_channels: Number of hidden channels in the TCN. Denoted as _H_ in the paper.
            skip_channels: Number of skip channels in the TCN. Denoted as _Sc_ in the paper.
            kernel_size: Kernel size in the TCN. Denoted as _P_ in the paper.
            layers: Number of layers in the TCN. Denoted as _X_ in the paper.
            repeats: Number of repeats in the TCN. Denoted as _R_ in the paper.
            causal: Whether to use causal convolutions in the TCN.
            norm: Normalization module to use in the TCN. If `None`, defaults to [LayerNorm][addse.layers.LayerNorm]
                with `causal=causal`. If a non-causal normalization module is provided, the TCN is not causal, even if
                `causal=True`.
        """
        super().__init__()
        hop_size = filter_size // 2 if hop_size is None else hop_size
        self.encoder = nn.Conv1d(1, num_filters, filter_size, stride=hop_size)
        self.decoder = nn.ConvTranspose1d(num_filters, 1, filter_size, stride=hop_size)
        self.tcn = ConvTasNetTCN(
            input_channels=input_channels * num_filters,
            output_channels=input_channels * output_channels * num_filters,
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            kernel_size=kernel_size,
            layers=layers,
            repeats=repeats,
            causal=causal,
            norm=functools.partial(LayerNorm, causal=causal) if norm is None else norm,
        )
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.hop_size = hop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        # pad to fit trailing samples in last encoder slice
        padding = (self.filter_size - x.shape[-1]) % self.hop_size
        x = F.pad(x, (0, padding))  # (B, C_in, t)
        x = x.reshape(-1, 1, x.shape[-1])  # (B * C_in, 1, t)
        x = self.encoder(x)  # (B * C_in, N, T)
        encoded = x.reshape(-1, self.input_channels, self.num_filters, x.shape[-1])  # (B, C_in, N, T)
        x = x.reshape(-1, self.input_channels * self.num_filters, x.shape[-1])  # (B, C_in * N, T)
        w = self.tcn(x)  # (B, C_in * C_out * N, T)
        w = w.reshape(w.shape[0], self.input_channels, self.output_channels, self.num_filters, -1)
        x = torch.einsum("bift,bijft->bjft", encoded, w)  # (B, C_out, N, T)
        x = x.reshape(-1, self.num_filters, x.shape[-1])  # (B * C_out, N, T)
        x = self.decoder(x)  # (B * C_out, 1, t)
        x = x.reshape(-1, self.output_channels, x.shape[-1])  # (B, C_out, t)
        return x[..., : x.shape[-1] - padding]


class ConvTasNetTCN(nn.Module):
    """Temporal convolutional network (TCN) used in Conv-TasNet."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bottleneck_channels: int,
        hidden_channels: int,
        skip_channels: int,
        kernel_size: int,
        layers: int,
        repeats: int,
        causal: bool,
        norm: Callable[[int], nn.Module],
    ) -> None:
        """Initialize the Conv-TasNet TCN."""
        super().__init__()
        self.norm = norm(input_channels)
        self.input_conv = nn.Conv1d(input_channels, bottleneck_channels, 1)
        self.blocks = nn.ModuleList(
            [
                ConvTasNetConv1DBlock(
                    input_channels=bottleneck_channels,
                    hidden_channels=hidden_channels,
                    skip_channels=skip_channels,
                    dilation=2**i,
                    kernel_size=kernel_size,
                    causal=causal,
                    last=b == repeats - 1 and i == layers - 1,
                    norm=norm,
                )
                for b in range(repeats)
                for i in range(layers)
            ]
        )
        self.activation = nn.PReLU()
        self.output_conv = nn.Conv1d(skip_channels, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.norm(x)
        x = self.input_conv(x)
        skips = 0
        for block in self.blocks:
            skip, x = block(x)
            skips += skip
        x = self.activation(skip)
        x = self.output_conv(x)
        return torch.sigmoid(x)


class ConvTasNetConv1DBlock(nn.Module):
    """1D convolutional block with PReLU activation and normalization used in Conv-TasNet."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        causal: bool,
        last: bool,
        norm: Callable[[int], nn.Module],
    ) -> None:
        """Initialize the Conv-TasNet 1D convolutional block."""
        super().__init__()
        self.conv = nn.Conv1d(input_channels, hidden_channels, 1)
        self.act_1 = nn.PReLU()
        self.norm_1 = norm(hidden_channels)
        self.dconv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, dilation=dilation, groups=hidden_channels)
        self.act_2 = nn.PReLU()
        self.norm_2 = norm(hidden_channels)
        self.skip_conv = nn.Conv1d(hidden_channels, skip_channels, 1)
        # the output of the residual convolution in the last block is not used
        self.res_conv = None if last else nn.Conv1d(hidden_channels, input_channels, 1)
        pad = (kernel_size - 1) * dilation
        self.pad = (pad, 0) if causal else (pad // 2, pad - pad // 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass."""
        h = self.conv(x)
        h = self.act_1(h)
        h = self.norm_1(h)
        h = F.pad(h, self.pad)
        h = self.dconv(h)
        h = self.act_2(h)
        h = self.norm_2(h)
        skip = self.skip_conv(h)
        res = None if self.res_conv is None else x + self.res_conv(h)
        return skip, res
