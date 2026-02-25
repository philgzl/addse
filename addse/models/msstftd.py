from collections.abc import Collection, Iterable

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from ..stft import STFT


class MSSTFTDiscriminator(nn.Module):
    """Multi-scale short-time Fourier transform (MS-STFT) discriminator."""

    def __init__(
        self,
        frame_lengths: Collection[int] = (127, 257, 509, 1021, 2053),
        hop_lengths: Collection[int | None] | None = None,
        n_ffts: Collection[int | None] | None = None,
        window: str = "flattop",
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: int = 32,
        kernel_size: tuple[int, int] = (9, 3),
        stride: tuple[int, int] = (2, 1),
        dilations: Iterable[int] = (1, 2, 4),
    ) -> None:
        """Initialize the MR-STFT discriminator."""
        super().__init__()
        hop_lengths = [None] * len(frame_lengths) if hop_lengths is None else hop_lengths
        n_ffts = [None] * len(frame_lengths) if n_ffts is None else n_ffts
        if len(hop_lengths) != len(frame_lengths):
            raise ValueError("`hop_lengths` must have the same length as `frame_lengths`.")
        if len(n_ffts) != len(frame_lengths):
            raise ValueError("`n_ffts` must have the same length as `frame_lengths`.")
        self.discriminators = nn.ModuleList(
            [
                STFTDiscriminator(
                    frame_length=frame_length,
                    hop_length=hop_length,
                    n_fft=n_fft,
                    window=window,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilations=dilations,
                )
                for frame_length, hop_length, n_fft in zip(frame_lengths, hop_lengths, n_ffts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """Forward pass."""
        outputs = []
        features = []
        for disc in self.discriminators:
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features


class STFTDiscriminator(nn.Module):
    """Short-time Fourier transform (STFT) discriminator."""

    def __init__(
        self,
        frame_length: int = 512,
        hop_length: int | None = None,
        n_fft: int | None = None,
        window: str = "flattop",
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: int = 32,
        kernel_size: tuple[int, int] = (9, 3),
        stride: tuple[int, int] = (2, 1),
        dilations: Iterable[int] = (1, 2, 4),
    ) -> None:
        """Initialize the STFT discriminator."""
        super().__init__()
        self.stft = STFT(frame_length=frame_length, hop_length=hop_length, n_fft=n_fft, window=window)
        self.convs = nn.ModuleList(
            [
                STFTDiscriminatorConv2d(2 * in_channels, num_channels, kernel_size),
                *[
                    STFTDiscriminatorConv2d(num_channels, num_channels, kernel_size, stride, dilation=(1, dilation))
                    for dilation in dilations
                ],
                STFTDiscriminatorConv2d(num_channels, num_channels, (kernel_size[1], kernel_size[1])),
                STFTDiscriminatorConv2d(num_channels, out_channels, (kernel_size[1], kernel_size[1]), activation=False),
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass."""
        x = self.stft(x)
        x = torch.cat([x.real, x.imag], dim=1)
        features = []
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        return x, features


class STFTDiscriminatorConv2d(nn.Module):
    """Short-time Fourier transform (STFT) discriminator 2D convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        activation: bool = True,
    ) -> None:
        """Initialize the STFT discriminator 2D convolutional layer."""
        super().__init__()
        self.conv = weight_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(((kernel_size[0] - 1) * dilation[0]) // 2, ((kernel_size[1] - 1) * dilation[1]) // 2),
                dilation=dilation,
            )
        )
        self.act = nn.LeakyReLU(0.2) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.conv(x))
