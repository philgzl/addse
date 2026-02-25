from collections.abc import Container, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.parametrizations import weight_norm


class ADM(nn.Module):
    """ADM similar to configuration F in EDM2 paper."""

    def __init__(
        self,
        num_channels: int = 1,
        base_channels: int = 96,
        num_res_blocks: int = 3,
        channel_mult: Sequence[int] = (1, 2, 3, 4),
        attn_levels: Container[int] = (),
    ) -> None:
        """Initialize ADM."""
        super().__init__()
        channels = [base_channels * mult for mult in channel_mult]
        ch_in, ch_emb = channels[0], channels[-1]
        self.emb_block = ADMEmbeddingBlock(ch_in, ch_emb)
        self.encoder = nn.ModuleList()
        ch_skip = []
        for i, ch_out in enumerate(channels):
            if i == 0:
                self.encoder.append(adm_conv2d(4 * num_channels + 1, ch_in, 3, 1, 1))
                ch_skip.append(ch_in)
            else:
                self.encoder.append(ADMBlock(ch_in, ch_in, ch_emb, "down", resample=True))
                ch_skip.append(ch_in)
            for _ in range(num_res_blocks):
                self.encoder.append(ADMBlock(ch_in, ch_out, ch_emb, "down", attn=i in attn_levels))
                ch_skip.append(ch_out)
                ch_in = ch_out
        self.decoder = nn.ModuleList()
        for i, ch_out in reversed(list(enumerate(channels))):
            if i == len(channels) - 1:
                self.decoder.append(ADMBlock(ch_in, ch_in, ch_emb, "up", attn=True))
                self.decoder.append(ADMBlock(ch_in, ch_in, ch_emb, "up"))
            else:
                self.decoder.append(ADMBlock(ch_in, ch_in, ch_emb, "up", resample=True))
            for _ in range(num_res_blocks + 1):
                self.decoder.append(ADMBlock(ch_in + ch_skip.pop(), ch_out, ch_emb, "up", attn=i in attn_levels))
                ch_in = ch_out
        self.out_conv = adm_conv2d(ch_in, 2 * num_channels, 3, 1, 1)
        self.downsampling_factor = 2 ** (len(channel_mult) - 1)
        self.num_res_blocks = num_res_blocks

    def forward(self, y: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass.

        Args:
            y: Complex-valued diffused speech tensor. Shape `(batch_size, num_channels, num_freqs, num_frames)`.
            x: Complex-valued noisy speech tensor. Shape `(batch_size, num_channels, num_freqs, num_frames)`.
            t: Diffusion step or noise level. Shape `(batch_size,)`.

        Returns:
            Complex-valued output score. Shape `(batch_size, num_channels, num_freqs, num_frames)`.
        """
        if x.shape != y.shape:
            raise ValueError(f"Input shapes must match. Got {x.shape} and {y.shape}.")
        if x.ndim != 4:
            raise ValueError(f"Input must be 4-dimensional. Got shape {x.shape}.")
        if x.shape[2] % self.downsampling_factor != 0 or x.shape[3] % self.downsampling_factor != 0:
            raise ValueError(
                f"Input size along dimensions 2 and 3 must be divisible by {self.downsampling_factor}. "
                f"Got shape {x.shape}."
            )
        emb = self.emb_block(t)
        x = torch.cat([x.real, x.imag, y.real, y.imag, torch.ones_like(x.real)], dim=1)
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x) if i == 0 else block(x, emb)
            skips.append(x)
        for i, block in enumerate(self.decoder):
            if i == 0 or i % (self.num_res_blocks + 2) == 1:
                x = block(x, emb)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        x = self.out_conv(x)
        return torch.complex(*x.chunk(2, dim=1))


class ADMBlock(nn.Module):
    """ADM block."""

    def __init__(
        self, in_ch: int, out_ch: int, emb_ch: int, kind: str, resample: bool = False, attn: bool = False
    ) -> None:
        """Initialize the ADM block."""
        super().__init__()
        self.resample = ADMResample(kind) if resample else nn.Identity()
        self.act = nn.SiLU()
        self.fc = weight_norm(nn.Linear(emb_ch, out_ch, bias=False))
        self.conv_1 = adm_conv2d(in_ch if kind == "up" else out_ch, out_ch, 3, 1, 1)
        self.conv_2 = adm_conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip_conv = adm_conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = ADMAttentionBlock(out_ch) if attn else nn.Identity()
        self.kind = kind

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Forward pass."""
        x = self.resample(x)
        if self.kind == "down":
            x = self.skip_conv(x)
            x = x / x.square().mean(dim=1, keepdim=True).sqrt().add(1e-4)  # pixel-norm
        h = self.conv_1(self.act(x))
        h *= 1 + self.fc(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv_2(self.act(h))
        if self.kind == "up":
            x = self.skip_conv(x)
        x = (x + h) / 2**0.5
        return self.attn(x)


class ADMAttentionBlock(nn.Module):
    """ADM attention block."""

    def __init__(self, num_channels: int) -> None:
        """Initialize the ADM attention block."""
        super().__init__()
        self.conv_qkv = adm_conv2d(num_channels, 3 * num_channels, 1)
        self.conv_out = adm_conv2d(num_channels, num_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        B, C, H, W = x.shape
        qkv = self.conv_qkv(x).reshape(B, 3 * C, H * W)
        qkv = qkv / qkv.square().mean(dim=1, keepdim=True).sqrt().add(1e-4)  # pixel-norm
        q, k, v = qkv.chunk(3, dim=1)
        h = torch.einsum("bci,bcj->bij", q, k) * C ** (-0.5)
        h = torch.softmax(h, dim=-1)
        h = torch.einsum("bij,bcj->bci", h, v).reshape(B, C, H, W)
        h = self.conv_out(h)
        return (x + h) / 2**0.5


class ADMResample(nn.Module):
    """ADM 2D resampling block."""

    kernel: Tensor

    def __init__(self, kind: str) -> None:
        """Initialize the ADM 2D resampling block."""
        super().__init__()
        assert kind in ("up", "down"), f"`kind` must be 'up' or 'down'. Got {kind}."
        self.kind = kind
        kernel = torch.tensor([1, 1], dtype=torch.float32)
        kernel = kernel.outer(kernel).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel / kernel.sum())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        kernel = (4 if self.kind == "up" else 1) * self.kernel.tile([x.shape[1], 1, 1, 1])
        func = F.conv_transpose2d if self.kind == "up" else F.conv2d
        return func(x, kernel, groups=x.shape[1], stride=2)


class ADMEmbeddingBlock(nn.Module):
    """ADM time step embedding block."""

    freqs: Tensor
    phases: Tensor

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the ADM time embedding block."""
        super().__init__()
        self.fc = weight_norm(nn.Linear(in_channels, out_channels, bias=False))
        self.act = nn.SiLU()
        self.register_buffer("freqs", 2 * torch.pi * torch.randn(in_channels))
        self.register_buffer("phases", 2 * torch.pi * torch.rand(in_channels))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = torch.cos(x[:, None] * self.freqs[None, :] + self.phases[None, :])
        return self.act(self.fc(x))


def adm_conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Conv2d:
    """2D convolutional layer with weight normalization and no bias."""
    return weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
