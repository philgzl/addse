import pytest
import torch

from addse.data import AudioStreamingDataset, DynamicMixingDataset


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(("segment_length", "max_length"), [(None, None), (1.0, None), (None, 1.1)])
@pytest.mark.parametrize("max_dynamic_range", [None, 100.0])
def test_audio_streaming_dset_repro(
    input_dir: str,
    shuffle: bool,
    segment_length: float | None,
    max_length: float | None,
    max_dynamic_range: float | None,
) -> None:
    """Test AudioStreamingDataset reproducibility."""
    dataset = AudioStreamingDataset(
        input_dir=input_dir,
        segment_length=segment_length,
        max_length=max_length,
        max_dynamic_range=max_dynamic_range,
        shuffle=shuffle,
    )
    data_1 = []
    for audio, _, _, _ in dataset:
        assert isinstance(audio, torch.Tensor)
        data_1.append(audio)
    data_2 = []
    for audio, _, _, _ in dataset:
        assert isinstance(audio, torch.Tensor)
        data_2.append(audio)
    assert all(torch.equal(a, b) for a, b in zip(data_1, data_2)) ^ shuffle


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(("segment_length", "max_length"), [(None, None), (1.0, None), (None, 1.1)])
@pytest.mark.parametrize("max_dynamic_range", [None, 100.0])
@pytest.mark.parametrize("break_at", [3])
def test_audio_streaming_dset_repro_partial(
    input_dir: str,
    shuffle: bool,
    segment_length: float | None,
    max_length: float | None,
    max_dynamic_range: float | None,
    break_at: int,
) -> None:
    """Test AudioStreamingDataset reproducibility with partial iteration."""
    dataset = AudioStreamingDataset(
        input_dir=input_dir,
        segment_length=segment_length,
        max_length=max_length,
        max_dynamic_range=max_dynamic_range,
        shuffle=shuffle,
    )
    data_1 = []
    for i, (audio, _, _, _) in enumerate(dataset):
        assert isinstance(audio, torch.Tensor)
        data_1.append(audio)
        if i == break_at:
            break
    assert i == break_at
    data_2 = []
    for i, (audio, _, _, _) in enumerate(dataset):
        assert isinstance(audio, torch.Tensor)
        data_2.append(audio)
        if i == break_at:
            break
    assert i == break_at
    assert all(torch.equal(a, b) for a, b in zip(data_1, data_2))


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("segment_length", [1.0])
@pytest.mark.parametrize("length", [None, 18])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("reset_rngs", [False, True])
def test_dynamic_dset_repro(
    input_dirs: tuple[str, str],
    shuffle: bool,
    segment_length: float | None,
    length: int | None,
    resume: bool,
    reset_rngs: bool,
) -> None:
    """Test DynamicMixingDataset reproducibility."""
    dataset = DynamicMixingDataset(
        AudioStreamingDataset(input_dirs[0], segment_length=segment_length, shuffle=shuffle),
        AudioStreamingDataset(input_dirs[1], segment_length=segment_length, shuffle=shuffle),
        length=length,
        resume=resume,
        reset_rngs=reset_rngs,
    )
    noisys = []
    cleans = []
    for noisy, clean, _ in dataset:
        assert isinstance(noisy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        noisys.append(noisy)
        cleans.append(clean)
    for i, (noisy, clean, _) in enumerate(dataset):
        assert isinstance(noisy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        assert torch.equal(noisy, noisys[i]) ^ (shuffle and (length is None or resume))
        assert torch.equal(clean, cleans[i]) ^ (shuffle and (length is None or resume))


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("segment_length", [1.0])
@pytest.mark.parametrize("length", [None, 18])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("reset_rngs", [False, True])
@pytest.mark.parametrize("break_at", [3])
def test_dynamic_dset_repro_partial(
    input_dirs: tuple[str, str],
    shuffle: bool,
    segment_length: float | None,
    length: int | None,
    resume: bool,
    reset_rngs: bool,
    break_at: int,
) -> None:
    """Test DynamicMixingDataset reproducibility with partial iteration."""
    dataset = DynamicMixingDataset(
        AudioStreamingDataset(input_dirs[0], segment_length=segment_length, shuffle=shuffle),
        AudioStreamingDataset(input_dirs[1], segment_length=segment_length, shuffle=shuffle),
        length=length,
        resume=resume,
        reset_rngs=reset_rngs,
    )
    noisys = []
    cleans = []
    for i, (noisy, clean, _) in enumerate(dataset):
        assert isinstance(noisy, torch.Tensor)
        assert isinstance(clean, torch.Tensor)
        noisys.append(noisy)
        cleans.append(clean)
        if i == break_at:
            break
    assert i == break_at
    for i, (noisy, clean, _) in enumerate(dataset):
        assert isinstance(noisy, torch.Tensor)
        assert torch.equal(noisy, noisys[i])
        assert isinstance(clean, torch.Tensor)
        assert torch.equal(clean, cleans[i])
        if i == break_at:
            break
    assert i == break_at
