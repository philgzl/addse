from typing import Any

import numpy as np
import pytest

from addse.metrics import (
    BaseMetric,
    DNSMOSMetric,
    LPSMetric,
    MCDMetric,
    NISQAMetric,
    PESQMetric,
    SBSMetric,
    SCOREQMetric,
    SDRMetric,
    STOIMetric,
    UTMOSMetric,
)


@pytest.fixture
def samples() -> tuple[np.ndarray, np.ndarray]:
    """Fixture for example samples."""
    generator = np.random.default_rng(0)
    shape = (2, 16000)
    return generator.standard_normal(shape), generator.standard_normal(shape)


def test_errors() -> None:
    """Test errors for invalid inputs."""
    metric = BaseMetric()
    with pytest.raises(TypeError, match="Inputs must be numpy arrays"):
        metric(None, None)
    with pytest.raises(ValueError, match="Inputs must be 2-dimensional"):
        metric(np.random.randn(32), np.random.randn(32))
    with pytest.raises(ValueError, match="Inputs must have the same shape"):
        metric(np.random.randn(1, 32), np.random.randn(2, 32))


@pytest.mark.parametrize(
    ("metric_cls", "kwargs"),
    [
        (SDRMetric, {}),
        (STOIMetric, {"fs": 16000}),
        (PESQMetric, {"fs": 16000}),
        (MCDMetric, {"fs": 16000}),
        (SCOREQMetric, {"fs": 16000}),
        (UTMOSMetric, {"fs": 16000}),
        (DNSMOSMetric, {"fs": 16000}),
        (LPSMetric, {"fs": 16000}),
        (NISQAMetric, {"fs": 16000}),
        (SBSMetric, {"fs": 16000}),
    ],
)
def test_metric(metric_cls: type[BaseMetric], kwargs: dict[str, Any], samples: tuple[np.ndarray, np.ndarray]) -> None:
    """Test a metric."""
    metric = metric_cls(**kwargs)
    output = metric(*samples)
    assert isinstance(output, float)
