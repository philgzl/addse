from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class MPDiscriminator(nn.Module):
    """Multi-period discriminator."""

    def __init__(
        self,
        periods: Iterable[int] = (2, 3, 5, 7, 11),
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        channels: Sequence[int] = (32, 128, 512, 1024, 1024),
        out_kernel_size: int = 3,
        out_stride: int = 1,
    ) -> None:
        """Initialize the multi-period discriminator."""
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                PDiscriminator(period, in_channels, kernel_size, stride, channels, out_kernel_size, out_stride)
                for period in periods
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


class PDiscriminator(nn.Module):
    """Period discriminator."""

    def __init__(
        self,
        period: int,
        in_channels: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        channels: Sequence[int] = (32, 128, 512, 1024, 1024),
        out_kernel_size: int = 3,
        out_stride: int = 1,
    ) -> None:
        """Initialize the period discriminator."""
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                PDiscriminatorConv1d(in_channels if i == 0 else channels[i - 1], channels[i], kernel_size, stride)
                for i in range(len(channels))
            ]
        )
        self.conv_out = PDiscriminatorConv1d(channels[-1], 1, out_kernel_size, out_stride, False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass."""
        features = []
        x = F.pad(x, (0, self.period - x.shape[-1] % self.period))
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // self.period, self.period)
        for conv in self.convs:
            x = conv(x)
            features.append(x)
        x = self.conv_out(x)
        features.append(x)
        return x, features


class PDiscriminatorConv1d(nn.Module):
    """Period discriminator 1D convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        """Initialize the period discriminator 1D convolutional layer."""
        super().__init__()
        self.conv = weight_norm(
            nn.Conv2d(
                in_channels, out_channels, (kernel_size, 1), stride=(stride, 1), padding=((kernel_size - 1) // 2, 0)
            )
        )
        self.act = nn.LeakyReLU(0.1) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.conv(x))
