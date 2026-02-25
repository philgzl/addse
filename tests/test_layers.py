from typing import Any

import pytest
import torch
import torch.nn as nn

from addse.layers import BatchNorm, GroupNorm, InstanceNorm, LayerNorm


@pytest.mark.parametrize(
    ("norm_cls", "kw"),
    [
        (GroupNorm, {"num_channels": 4, "num_groups": 2}),
        (InstanceNorm, {"num_channels": 4}),
        (LayerNorm, {"num_channels": 4}),
    ],
)
@pytest.mark.parametrize("shape", [(4, 4)])
def test_norm_error(
    torch_rng: torch.Generator, norm_cls: type[nn.Module], kw: dict[str, Any], shape: tuple[int, ...]
) -> None:
    """Test that normalization modules raise an error when input has incorrect shape."""
    x = torch.randn(shape, generator=torch_rng)
    norm = norm_cls(**kw).eval()
    with pytest.raises(AssertionError):
        norm(x)


@pytest.mark.parametrize(
    ("norm_cls", "kw", "shape", "dim"),
    [
        (InstanceNorm, {"num_channels": 4}, (4, 4, 32, 32), (2, 3)),
        (InstanceNorm, {"num_channels": 4}, (4, 4, 32), (2,)),
        (LayerNorm, {"num_channels": 4}, (4, 4, 32, 32), (1, 2, 3)),
        (LayerNorm, {"num_channels": 4}, (4, 4, 32), (1, 2)),
    ],
)
def test_causal_norm_output_value(
    torch_rng: torch.Generator,
    norm_cls: type[nn.Module],
    kw: dict[str, Any],
    shape: tuple[int, ...],
    dim: tuple[int, ...],
) -> None:
    """Test the output value of causal normalization modules."""
    x = torch.randn(shape, generator=torch_rng)
    norm = norm_cls(causal=True, **kw).eval()
    y = norm(x)
    assert y.shape == x.shape
    for i in range(1, x.shape[-1]):
        mean = x[..., : i + 1].mean(dim, keepdim=True)
        std = x[..., : i + 1].std(dim, keepdim=True, correction=0)
        expected = (x[..., [i]] - mean) / std
        assert torch.allclose(y[..., [i]], expected, atol=1e-4), (y[..., [i]] - expected).abs().max()


@pytest.mark.parametrize(
    ("norm_cls", "kw", "eval", "causal"),
    [
        (GroupNorm, {"num_channels": 4, "causal": True, "num_groups": 2}, True, True),
        (InstanceNorm, {"num_channels": 4, "causal": True}, True, True),
        (LayerNorm, {"num_channels": 4, "causal": True, "center": False}, True, True),
        (LayerNorm, {"num_channels": 4, "causal": True, "center": True}, True, True),
        (LayerNorm, {"num_channels": 4, "element_wise": True, "center": False}, True, True),
        (LayerNorm, {"num_channels": 4, "element_wise": True, "center": True}, True, True),
        (LayerNorm, {"num_channels": 4, "frame_wise": True, "center": False}, True, True),
        (LayerNorm, {"num_channels": 4, "frame_wise": True, "center": True}, True, True),
        (BatchNorm, {"num_channels": 4}, True, True),
        (BatchNorm, {"num_channels": 4}, False, False),
    ],
)
@pytest.mark.parametrize("shape", [(2, 4, 16, 32), (2, 4, 32)])
def test_norm_causality(
    torch_rng: torch.Generator,
    norm_cls: type[nn.Module],
    kw: dict[str, Any],
    eval: bool,
    causal: bool,
    shape: tuple[int, ...],
) -> None:
    """Test the causality of normalization modules."""
    norm = norm_cls(**kw)
    if eval:
        norm.eval()
    for i in range(1, shape[-1]):
        x = torch.randn(shape, generator=torch_rng)
        x[..., i] = float("nan")
        y = norm(x)
        assert y.shape == x.shape
        if causal:
            assert not y[..., :i].isnan().any()
            assert y[..., i].isnan().all()
        else:
            assert y.isnan().all()
