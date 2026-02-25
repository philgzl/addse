import pytest
import torch

from addse.data import AudioStreamingDataLoader, AudioStreamingDataset, DynamicMixingDataset


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("segment_length", [1.0])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_workers", [0, 4])
def test_dataloader_shuffle_with_audio_streaming_dataset(
    input_dir: str, shuffle: bool, segment_length: float, batch_size: int, num_workers: int
) -> None:
    """Test AudioStreamingDataLoader shuffling with AudioStreamingDataset."""
    dataset = AudioStreamingDataset(input_dir=input_dir, segment_length=segment_length, shuffle=shuffle)
    dataloader = AudioStreamingDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    epoch_1 = []
    for audio, _, _, _ in dataloader:
        assert isinstance(audio, torch.Tensor)
        assert audio.shape[0] <= batch_size
        epoch_1.append(audio)
    epoch_2 = []
    for audio, _, _, _ in dataloader:
        assert isinstance(audio, torch.Tensor)
        assert audio.shape[0] <= batch_size
        epoch_2.append(audio)
    assert len(epoch_1) == len(epoch_2)
    assert all(torch.equal(a, b) for a, b in zip(epoch_1, epoch_2)) ^ shuffle


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("segment_length", [1.0])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("break_at", [3])
def test_dataloader_shuffle_with_audio_streaming_dataset_and_partial_iteration(
    input_dir: str, shuffle: bool, segment_length: float, batch_size: int, num_workers: int, break_at: int
) -> None:
    """Test AudioStreamingDataLoader shuffling with AudioStreamingDataset and partial iteration."""
    dataset = AudioStreamingDataset(input_dir=input_dir, segment_length=segment_length, shuffle=shuffle)
    dataloader = AudioStreamingDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    epoch_1 = []
    for i, (audio, _, _, _) in enumerate(dataloader):
        assert isinstance(audio, torch.Tensor)
        assert audio.shape[0] <= batch_size
        epoch_1.append(audio)
        if i == break_at:
            break
    assert i == break_at
    epoch_2 = []
    for i, (audio, _, _, _) in enumerate(dataloader):
        assert isinstance(audio, torch.Tensor)
        assert audio.shape[0] <= batch_size
        epoch_2.append(audio)
        if i == break_at:
            break
    assert i == break_at
    assert len(epoch_1) == len(epoch_2)
    assert all(torch.equal(a, b) for a, b in zip(epoch_1, epoch_2)) ^ shuffle


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("reset_rngs", [False, True])
@pytest.mark.parametrize("segment_length", [1.0])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("length", [None, 18])
@pytest.mark.parametrize("resume", [False, True])
def test_dataloader_shuffle_with_dynamic_mixing_dataset(
    input_dirs: tuple[str, str],
    shuffle: bool,
    reset_rngs: bool,
    segment_length: float,
    batch_size: int,
    num_workers: int,
    length: int | None,
    resume: bool,
) -> None:
    """Test AudioStreamingDataLoader shuffling with DynamicMixingDataset."""
    dataset = DynamicMixingDataset(
        AudioStreamingDataset(input_dirs[0], segment_length=segment_length, shuffle=shuffle),
        AudioStreamingDataset(input_dirs[1], segment_length=segment_length, shuffle=shuffle),
        length=length,
        resume=resume,
        reset_rngs=reset_rngs,
    )
    dataloader = AudioStreamingDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    epoch_1_noisy = []
    epoch_1_clean = []
    for noisy, clean, _ in dataloader:
        assert isinstance(noisy, torch.Tensor)
        assert noisy.shape[0] <= batch_size
        epoch_1_noisy.append(noisy)
        assert isinstance(clean, torch.Tensor)
        assert clean.shape[0] <= batch_size
        epoch_1_clean.append(clean)
    epoch_2_noisy = []
    epoch_2_clean = []
    for noisy, clean, _ in dataloader:
        assert isinstance(noisy, torch.Tensor)
        assert noisy.shape[0] <= batch_size
        epoch_2_noisy.append(noisy)
        assert isinstance(clean, torch.Tensor)
        assert clean.shape[0] <= batch_size
        epoch_2_clean.append(clean)
    assert len(epoch_1_noisy) == len(epoch_2_noisy)
    assert len(epoch_1_clean) == len(epoch_2_clean)
    assert all(torch.equal(a, b) for a, b in zip(epoch_1_noisy, epoch_2_noisy)) ^ (
        not (length is None and not shuffle and reset_rngs or length is not None and not resume)
    )
    assert all(torch.equal(a, b) for a, b in zip(epoch_1_clean, epoch_2_clean)) ^ (
        not (length is None and not shuffle or length is not None and not resume)
    )


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("reset_rngs", [False, True])
@pytest.mark.parametrize("segment_length", [1.0])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_workers", [0, 4])
@pytest.mark.parametrize("length", [None, 14])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("break_at", [3])
def test_dataloader_shuffle_with_dynamic_mixing_dataset_and_partial_iteration(
    input_dirs: tuple[str, str],
    shuffle: bool,
    reset_rngs: bool,
    segment_length: float,
    batch_size: int,
    num_workers: int,
    length: int | None,
    resume: bool,
    break_at: int,
) -> None:
    """Test AudioStreamingDataLoader shuffling with DynamicMixingDataset and partial iteration."""
    dataset = DynamicMixingDataset(
        AudioStreamingDataset(input_dirs[0], segment_length=segment_length, shuffle=shuffle),
        AudioStreamingDataset(input_dirs[1], segment_length=segment_length, shuffle=shuffle),
        length=length,
        resume=resume,
        reset_rngs=reset_rngs,
    )
    dataloader = AudioStreamingDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    epoch_1_noisy = []
    epoch_1_clean = []
    for i, (noisy, clean, _) in enumerate(dataloader):
        assert isinstance(noisy, torch.Tensor)
        assert noisy.shape[0] <= batch_size
        epoch_1_noisy.append(noisy)
        assert isinstance(clean, torch.Tensor)
        assert clean.shape[0] <= batch_size
        epoch_1_clean.append(clean)
        if i == break_at:
            break
    assert i == break_at
    epoch_2_noisy = []
    epoch_2_clean = []
    for i, (noisy, clean, _) in enumerate(dataloader):
        assert isinstance(noisy, torch.Tensor)
        assert noisy.shape[0] <= batch_size
        epoch_2_noisy.append(noisy)
        assert isinstance(clean, torch.Tensor)
        assert clean.shape[0] <= batch_size
        epoch_2_clean.append(clean)
        if i == break_at:
            break
    assert i == break_at
    assert len(epoch_1_noisy) == len(epoch_2_noisy)
    assert len(epoch_1_clean) == len(epoch_2_clean)
    assert all(torch.equal(a, b) for a, b in zip(epoch_1_noisy, epoch_2_noisy)) ^ (
        not (length is None and not shuffle and reset_rngs or length is not None and not resume)
    )
    assert all(torch.equal(a, b) for a, b in zip(epoch_1_clean, epoch_2_clean)) ^ (
        not (length is None and not shuffle or length is not None and not resume)
    )
