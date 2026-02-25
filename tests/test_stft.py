import pytest
import torch

from addse.stft import STFT


@pytest.mark.parametrize(
    ("frame_length", "hop_length", "n_fft", "window"),
    [
        (512, 256, None, "hann"),
        (256, 128, None, "hann"),
        (512, 256, 1024, "hann"),
        (256, 128, 1024, "hann"),
        (510, 128, 1024, "hann"),
        (512, 256, None, "boxcar"),
        (256, 256, None, "boxcar"),
        (512, 256, 1024, "boxcar"),
        (256, 256, 1024, "boxcar"),
    ],
)
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("shape", [(1000,), (2, 1000), (2, 3, 1000)])
def test_stft_reconstruction(
    torch_rng: torch.Generator,
    frame_length: int,
    hop_length: int,
    n_fft: int | None,
    window: str,
    norm: bool,
    shape: tuple[int, ...],
) -> None:
    """Test STFT reconstruction."""
    batch = torch.randn(shape, generator=torch_rng)
    stft = STFT(frame_length=frame_length, hop_length=hop_length, n_fft=n_fft, window=window, norm=norm)
    output = stft(batch)
    output = stft.inverse(output, n=batch.shape[-1])
    assert torch.allclose(batch, output, atol=1e-5), (batch - output).abs().max()
