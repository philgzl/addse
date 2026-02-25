import pytest
import torch

from addse.losses import BaseLoss, MelSpecLoss, MultiTaskLoss, SDRLoss


@pytest.fixture
def samples() -> tuple[torch.Tensor, torch.Tensor]:
    """Fixture for example samples."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, 2, 32, generator=generator)
    y = torch.randn(4, 2, 32, generator=generator)
    return x, y


def check_output(loss: BaseLoss, samples: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Check that the output loss dictionary is valid."""
    x, y = samples
    output = loss(x, y)
    assert isinstance(output, dict)
    assert "loss" in output
    for key, value in output.items():
        assert isinstance(value, torch.Tensor), f"Value for key '{key}' is not a torch tensor."
        assert value.ndim == 0, f"Value for key '{key}' is not a scalar."
        assert value.numel() == 1, f"Value for key '{key}' is not a single element."


def check_sample_independence(loss: BaseLoss, samples: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Check that the loss for each sample does not depend on other samples in the batch."""
    x, y = samples
    output_1 = loss(x, y)["loss"]
    output_2 = 0.0
    for i in range(x.shape[0]):
        single_x = x[i : i + 1]
        single_y = y[i : i + 1]
        output_2 += loss(single_x, single_y)["loss"]
    assert torch.allclose(output_1, output_2 / x.shape[0])


def run_loss_tests(loss: BaseLoss, samples: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Run standard tests for a loss function."""
    check_output(loss, samples)
    check_sample_independence(loss, samples)


def test_errors() -> None:
    """Test errors for invalid inputs."""
    loss = BaseLoss()
    with pytest.raises(TypeError, match="Inputs must be torch tensors"):
        loss(None, None)
    with pytest.raises(ValueError, match="Inputs must be 3-dimensional"):
        loss(torch.randn(1, 32), torch.randn(1, 32))
    with pytest.raises(ValueError, match="Inputs must have the same shape"):
        loss(torch.randn(1, 1, 32), torch.randn(2, 1, 32))


def test_multi_task_loss(samples: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test the multi-task loss."""
    loss = MultiTaskLoss([MelSpecLoss(), SDRLoss()], [1.0, 1.0])
    run_loss_tests(loss, samples)


@pytest.mark.parametrize("scale_invariant", [False, True])
@pytest.mark.parametrize("zero_mean", [False, True])
def test_sdr_loss(samples: tuple[torch.Tensor, torch.Tensor], scale_invariant: bool, zero_mean: bool) -> None:
    """Test the SDR loss."""
    loss = SDRLoss(scale_invariant=scale_invariant, zero_mean=zero_mean)
    run_loss_tests(loss, samples)


def test_mel_spectrogram_loss(samples: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test the mel-spectrogram loss."""
    loss = MelSpecLoss()
    run_loss_tests(loss, samples)
