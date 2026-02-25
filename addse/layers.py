from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: rename `causal` to `cumulative` in all normalization modules and functions


class GroupNorm(nn.Module):
    r"""Group normalization.

    Input tensors must have shape `(B, C, ...)` where `B` is the batch dimension, `C` is the channel dimension, and
    `...` are the spatial dimensions (e.g. height and width in computer vision, frequency and time in audio, or sequence
    length in NLP). The statistics are aggregated over grouped channels and spatial dimensions as in [^1], Figure 2.
    Namely,

    $$y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}} (1 + \gamma) + \beta,$$

    where $\gamma$ and $\beta$ are channel-specific learnable scale and shift parameters. Note the reparameterization of
    the scale parameter compared to the default PyTorch implementation.

    [^1]: Y. Wu and K. He, "Group normalization", ECCV, 2018.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, causal: bool = False) -> None:
        """Initialize the group normalization module.

        Args:
            num_groups: Number of groups to separate the channels into.
            num_channels: Number of channels in input tensors.
            eps: Small value for numerical stability.
            causal: If `True`, normalization statistics are cumulatively aggregated along the time dimension. The time
                dimension must be the last dimension of the input tensor.
        """
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"`num_channels` must be divisible by `num_groups`, got {num_channels} and {num_groups}.")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.causal = causal
        self.weight = nn.Parameter(torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        assert x.ndim >= 3, f"{type(self).__name__} input must be at least 3-dimensional, got shape {x.shape}."
        assert x.shape[1] == self.num_channels, (
            f"Expected {type(self).__name__} input to have {self.num_channels} channels, got {x.shape[1]}."
        )
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps, self.causal, False)


class LayerNorm(nn.Module):
    r"""Layer normalization.

    Input tensors must have shape `(B, C, ...)` where `B` is the batch dimension, `C` is the channel dimension, and
    `...` are the spatial dimensions (e.g. height and width in computer vision, frequency and time in audio, or sequence
    length in NLP). Namely,

    $$y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}} (1 + \gamma) + \beta,$$

    where $\gamma$ and $\beta$ are channel-specific learnable scale and shift parameters. Note the reparameterization of
    the scale parameter compared to the default PyTorch implementation.

    If `element_wise` and `frame_wise` are both `False`, the statistics are aggregated over the channel dimension and
    all spatial dimensions as in [^1], Figure 2. In this case, setting `causal=False` matches the global layer
    normalization in [^2], while setting `causal=True` matches the cumulative layer normalization in [^2]. The time
    dimension must be the last dimension of input tensors.

    If `element_wise` is `True`, the statistics are aggregated over the channel dimension only as in [^3]. I.e. each
    element (e.g. pixel in computer vision, time-frequency unit in audio, or token in NLP) is normalized independently.

    If `frame_wise` is `True`, the statistics are aggregated over the channel dimension and all spatial dimensions
    except the time dimension. The time dimension must be the last dimension of input tensors.

    [^1]: Y. Wu and K. He, "Group normalization", ECCV, 2018.
    [^2]:
        Y. Luo and N. Mesgarani, "Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation",
        in IEEE/ACM TASLP, 2019.
    [^3]:
        S. Shen, Z. Yao, A. Gholami, M. W. Mahoney, and K. Keutzer, "PowerNorm: Rethinking batch normalization in
        transformers", ICML, 2020.
    """

    def __init__(
        self,
        num_channels: int,
        element_wise: bool = False,
        frame_wise: bool = False,
        causal: bool = False,
        center: bool = True,
        eps: float = 1e-5,
    ) -> None:
        r"""Initialize the layer normalization module.

        Args:
            num_channels: Number of channels in input tensors.
            element_wise: If `True`, each element (e.g. pixel in computer vision, time-frequency unit in audio, or token
                in NLP) is normalized independently. Mutually exclusive with `frame_wise` and `causal`.
            frame_wise: If `True`, each time frame is normalized independently. The time dimension must be the last
                dimension of input tensors. Mutually exclusive with `element_wise` and `causal`.
            causal: If `True`, normalization statistics are cumulatively aggregated along the time dimension. The time
                dimension must be the last dimension of the input tensor. Mutually exclusive with `element_wise` and
                `frame_wise`.
            center: If `False`, the mean is not subtracted from the input, and the input is scaled using the root mean
                square (RMS) instead of the variance. The bias term $\beta$ is also omitted.
            eps: Small value for numerical stability.
        """
        super().__init__()
        if element_wise + frame_wise + causal > 1:
            raise ValueError(
                "Only one of `element_wise`, `frame_wise`, and `causal` can be `True`. "
                f"Got {element_wise}, {frame_wise}, and {causal}."
            )
        self.num_channels = num_channels
        self.element_wise = element_wise
        self.frame_wise = frame_wise
        self.causal = causal
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels)) if center else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        assert x.ndim >= 3, f"{type(self).__name__} input must be at least 3-dimensional, got shape {x.shape}."
        assert x.shape[1] == self.num_channels, (
            f"Expected {type(self).__name__} input to have {self.num_channels} channels, got {x.shape[1]}."
        )
        if self.element_wise:
            x = x.moveaxis(1, -1)
            if self.bias is not None:  # if centering
                x = F.layer_norm(x, x.shape[-1:], 1 + self.weight, self.bias, self.eps)
            else:
                x = F.rms_norm(x, x.shape[-1:], 1 + self.weight, self.eps)
            return x.moveaxis(-1, 1)
        return group_norm(x, 1, self.weight, self.bias, self.eps, self.causal, self.frame_wise)


class InstanceNorm(GroupNorm):
    r"""Instance normalization.

    Input tensors must have shape `(B, C, ...)` where `B` is the batch dimension, `C` is the channel' dimension, and
    `...` are the spatial dimensions (e.g. height and width in computer vision, frequency and time in audio, or sequence
    length in NLP). The statistics are aggregated over the spatial dimensions as in [^1], Figure 2. Namely,

    $$y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}} (1 + \gamma) + \beta,$$

    where $\gamma$ and $\beta$ are channel-specific learnable scale and shift parameters. Note the reparameterization of
    the scale parameter compared to the default PyTorch implementation.

    [^1]: Y. Wu and K. He, "Group normalization", ECCV, 2018.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, causal: bool = False) -> None:
        """Initialize the instance normalization module.

        Args:
            num_channels: Number of channels in input tensors.
            eps: Small value for numerical stability.
            causal: If `True`, normalization statistics are cumulatively aggregated along the time dimension. The time
                dimension must be the last dimension of the input tensor.
        """
        super().__init__(num_groups=num_channels, num_channels=num_channels, eps=eps, causal=causal)


class BatchNorm(nn.Module):
    r"""Batch normalization.

    Input tensors must have shape `(B, C, ...)` where `B` is the batch dimension, `C` is the channel dimension, and
    `...` are the spatial dimensions (e.g. height and width in computer vision, frequency and time in audio, or sequence
    length in NLP). The statistics are aggregated over the batch and spatial dimensions as in [^1], Figure 2. Namely,

    $$y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathbb{V}[x] + \epsilon}} (1 + \gamma) + \beta,$$

    where $\gamma$ and $\beta$ are channel-specific learnable scale and shift parameters. Note the reparameterization of
    the scale parameter compared to the default PyTorch implementation.

    Unlike other normalization modules, this module has `track_running_stats` and `momentum` options.

    [^1]: Y. Wu and K. He, "Group normalization", ECCV, 2018.
    """

    running_mean: torch.Tensor | None
    running_var: torch.Tensor | None
    num_batches_tracked: torch.Tensor | None

    def __init__(
        self, num_channels: int, eps: float = 1e-5, track_running_stats: bool = True, momentum: float | None = 0.1
    ) -> None:
        """Initialize the batch normalization module.

        Args:
            num_channels: Number of channels in input tensors.
            eps: Small value for numerical stability.
            track_running_stats: If `True`, normalization statistics are aggregated over batches during training and
                saved for evaluation. If `False`, statistics are computed from the current batch both during training
                and evaluation.
            momentum: Momentum for running statistics. The bigger the value, the more weight is given to the current
                batch statistics. Ignored if `track_running_stats` is `False`. If `None`, running statistics are
                cumulatively aggregated over batches without decay.
        """
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.weight = nn.Parameter(torch.zeros(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_channels))
            self.register_buffer("running_var", torch.ones(num_channels))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        assert x.ndim >= 2, f"{type(self).__name__} input must be at least 2-dimensional, got shape {x.shape}."
        assert x.shape[1] == self.num_channels, (
            f"Expected {type(self).__name__} input to have {self.num_channels} channels, got {x.shape[1]}."
        )
        momentum = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            momentum = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum
        use_current_batch_stats = self.training or self.running_mean is None and self.running_var is None
        mean = self.running_mean if not self.training or self.track_running_stats else None
        var = self.running_var if not self.training or self.track_running_stats else None
        h = x.reshape(x.shape[0], x.shape[1], -1)
        h = F.batch_norm(h, mean, var, 1 + self.weight, self.bias, use_current_batch_stats, momentum, self.eps)
        return h.reshape(x.shape)


class BandSplit(nn.Module):
    """Band-split module."""

    def __init__(
        self,
        subband_idx: Iterable[tuple[int, int]],
        input_channels: int,
        output_channels: int,
        norm: Callable[[int], nn.Module],
    ) -> None:
        """Initialize the band-split module."""
        super().__init__()
        self.subband_idx = subband_idx
        self.norm = nn.ModuleList([norm(2 * input_channels * (end - start)) for start, end in subband_idx])
        self.fc = nn.ModuleList(
            [nn.Conv1d(2 * input_channels * (end - start), output_channels, 1) for start, end in subband_idx]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Complex-valued short-time Fourier transform. Shape `(batch_size, input_channels, num_freqs, num_frames)`.

        Returns:
            Output tensor with shape `(batch_size, output_channels, num_bands, num_frames)`.
        """
        assert x.ndim == 4, f"Input must be 4-dimensional, got shape {x.shape}."
        B, _, _, T = x.shape
        x = torch.view_as_real(x)  # (B, C_in, F, T, 2)
        out = []
        for i, (start, end) in enumerate(self.subband_idx):
            h = x[:, :, start:end, :, :]  # (B, C_in, end - start, T, 2)
            h = h.moveaxis(-1, 1).reshape(B, -1, T)  # (B, 2 * C * (end - start), T)
            h = self.norm[i](h)  # (B, 2 * C * (end - start), T)
            h = self.fc[i](h)  # (B, N, T)
            out.append(h)
        return torch.stack(out, dim=2)  # (B, N, K, T)


class BandMerge(nn.Module):
    """Band-merge module."""

    def __init__(
        self,
        subband_idx: Iterable[tuple[int, int]],
        input_channels: int,
        output_channels: int,
        num_channels: int,
        norm: Callable[[int], nn.Module],
        mlp: Callable[[int, int, Callable[[int], nn.Module]], nn.Module],
        residual: bool,
    ) -> None:
        """Initialize the band-merge module."""
        super().__init__()
        self.mlp_mask = nn.ModuleList(
            [
                mlp(num_channels, 2 * input_channels * output_channels * (end - start), norm)
                for start, end in subband_idx
            ]
        )
        self.mlp_res = (
            nn.ModuleList([mlp(num_channels, 2 * output_channels * (end - start), norm) for start, end in subband_idx])
            if residual
            else None
        )
        self.input_channels = input_channels
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: Input tensor with shape `(batch_size, input_channels, num_bands, num_frames)`.

        Returns:
            Tuple `(mask, residual)` where `mask` are complex-valued spatial filtering coefficients with shape
            `(batch_size, input_channels, output_channels, num_freqs, num_frames)`, and `residual` is a residual
            additive short-time Fourier transform with shape `(batch_size, output_channels, num_freqs, num_frames)` or
            `None` if `residual=False`.
        """
        B, _, K, T = x.shape
        submasks = []
        for i in range(K):
            h = x[:, :, i, :]  # (B, N, T)
            submask = self.mlp_mask[i](h)  # (B, 2 * C_in * C_out * (end - start), T)
            submask = submask.reshape(B, 2, self.input_channels, self.output_channels, -1, T)
            submasks.append(submask)
        mask = torch.cat(submasks, dim=-2)  # (B, 2, C_in, C_out, F, T)
        mask = torch.complex(*mask.unbind(1))  # (B, C_in, C_out, F, T)
        if self.mlp_res is None:
            return mask, None
        subresiduals = []
        for i in range(K):
            h = x[:, :, i, :]  # (B, N, T)
            subresidual = self.mlp_res[i](h)  # (B, 2 * C_out * (end - start), T)
            subresidual = subresidual.reshape(B, 2, self.output_channels, -1, T)
            subresiduals.append(subresidual)
        residual = torch.cat(subresiduals, dim=-2)  # (B, 2, C_out, F, T)
        residual = torch.complex(*residual.unbind(1))  # (B, C_out, F, T)
        return mask, residual


def group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    causal: bool,
    frame_wise: bool,
) -> torch.Tensor:
    """Functional interface for group normalization.

    See [GroupNorm][addse.layers.GroupNorm] for details.
    """
    if causal and frame_wise:
        raise ValueError("`causal` and `frame_wise` cannot both be `True`.")
    if causal:
        h = x.reshape(x.shape[0], num_groups, x.shape[1] // num_groups, -1, x.shape[-1])
        count = torch.arange(1, h.shape[-1] + 1, device=h.device)
        mu_2 = h.pow(2).mean((2, 3), keepdim=True).cumsum(-1) / count
        if bias is not None:  # if centering
            mu_1 = h.mean((2, 3), keepdim=True).cumsum(-1) / count
            var = mu_2 - mu_1.pow(2)
            h = (h - mu_1) / (var + eps).sqrt()
            h = h.reshape(x.shape[0], x.shape[1], -1)
            h = h * (1 + weight).unsqueeze(1) + bias.unsqueeze(1)
        else:
            h = h / (mu_2 + eps).sqrt()
            h = h.reshape(x.shape[0], x.shape[1], -1)
            h = h * (1 + weight).unsqueeze(1)
        return h.reshape(x.shape)
    h = x.moveaxis(-1, 1).flatten(0, 1) if frame_wise else x
    if bias is not None:  # if centering
        h = F.group_norm(h, num_groups, (1 + weight), bias, eps)
    else:
        batch_size, num_channels, *spatial_dims = h.shape
        h = h.reshape(batch_size, num_groups, num_channels // num_groups, -1)
        mu_2 = h.pow(2).mean((2, 3), keepdim=True)
        h = h / (mu_2 + eps).sqrt()
        h = h.reshape(batch_size, num_channels, -1)
        h = h * (1 + weight).unsqueeze(1)
        h = h.reshape(batch_size, num_channels, *spatial_dims)
    return h.unflatten(0, (x.shape[0], x.shape[-1])).moveaxis(1, -1) if frame_wise else h
