import functools
import io
import logging
import warnings
from collections.abc import Iterator
from typing import Any

import litdata as ld
import soundfile as sf
import soxr
import torch
from litdata.utilities.base import __NUM_SAMPLES_YIELDED_KEY__, __SAMPLES_KEY__

from .utils import dynamic_range, set_snr

ASDOutput = tuple[torch.Tensor, int, str, int]

logger = logging.getLogger(__name__)


class AudioStreamingDataset(ld.StreamingDataset):
    """Audio streaming dataset."""

    def __init__(
        self,
        input_dir: str,
        fs: int | None = None,
        segment_length: float | None = None,
        max_length: float | None = None,
        max_dynamic_range: float | None = None,
        shuffle: bool = False,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize the audio streaming dataset.

        Args:
            input_dir: Path or URL to LitData-optimized audio data.
            fs: Optional sample rate to resample to.
            segment_length: Audio segment length in seconds. If provided, audio files are concatenated and segmented
                into chunks of this length. Else, audio files are yielded as is and may have variable length.
            max_length: Maximum output length in seconds. If provided, audio files longer than this are skipped. Cannot
                be used together with `segment_length`.
            max_dynamic_range: Maximum dynamic range in dB. If provided, audio files and segments with a dynamic range
                greater than this value are skipped.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.
            **kwargs: Additional keyword arguments passed to parent constructor.
        """
        super().__init__(input_dir, shuffle=shuffle, seed=seed + 42, **kwargs)

        if segment_length is not None and max_length is not None:
            raise ValueError("`max_length` cannot be used with `segment_length`.")

        self.segment_length = segment_length
        self.fs = fs
        self.max_length = max_length
        self.max_dynamic_range = max_dynamic_range
        self._fs = None
        self._queue = None

    def __iter__(self) -> Iterator[ASDOutput]:
        """Iterate over the dataset."""
        self._queue = None
        return super().__iter__()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        """Get an item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            Audio data with shape `(1, num_samples)` and name.
        """
        item = super().__getitem__(index)
        if isinstance(item, dict) and "audio" in item and "name" in item:
            bytes, name = item["audio"], item["name"]
        elif isinstance(item, tuple) and len(item) == 2:
            bytes, name = item
        else:
            raise ValueError(
                "Invalid dataset item format. "
                "Expected a dict with 'audio' and 'name' keys, or a tuple of (bytes, name). "
                f"Got {type(item).__name__}"
                f"{f' with length {len(item)}.' if isinstance(item, (tuple)) else ''}"
                f"{f' with keys {list(item.keys())}.' if isinstance(item, dict) else ''}"
                f"{'.' if not isinstance(item, dict | tuple) else ''}"
            )
        buffer = io.BytesIO(bytes)
        x, fs = sf.read(buffer, dtype="float32", always_2d=True)
        if x.shape[1] > 1:
            # TODO: make channel selection configurable instead of always picking the first channel
            x = x[:, [0]]
        if self.fs is not None and fs != self.fs:
            x = soxr.resample(x, fs, self.fs)
            fs = self.fs
        self._fs = fs if self._fs is None else self._fs
        assert fs == self._fs, "All files must have the same sample rate."
        logger.debug(name)
        return torch.from_numpy(x.T), name

    def __next__(self) -> ASDOutput:
        """Get the next item from the dataset.

        Returns:
            Audio data with shape `(1, num_samples)`, sample rate, name, and number of files loaded to get this item.
            The number of files loaded is required by [DynamicMixingDataset][addse.data.DynamicMixingDataset].
        """
        if self.segment_length is None:
            x, name = super().__next__()
            files_loaded = 1
            while not self.check(x, name):
                x, name = super().__next__()
                files_loaded += 1
            assert isinstance(self._fs, int)
            return x, self._fs, name, files_loaded
        segment, fs, name, files_loaded = self.next_segment()
        while not self.check(segment, name):
            segment, fs, name, more_files_loaded = self.next_segment()
            files_loaded += more_files_loaded
        return segment, fs, name, files_loaded

    def next_segment(self) -> ASDOutput:
        """Get the next audio segment from the dataset."""
        files_loaded = 0
        while self._queue is None or self._queue.shape[-1] < int(self.segment_length * self._fs):
            x, name = super().__next__()
            files_loaded += 1
            while not self.check(x, name):
                x, name = super().__next__()
                files_loaded += 1
            self._queue = x if self._queue is None else torch.cat([self._queue, x], dim=-1)
        segment = self._queue[:, : int(self.segment_length * self._fs)]
        self._queue = self._queue[:, int(self.segment_length * self._fs) :]
        return segment, self._fs, "segment", files_loaded  # TODO: derive concatenated names

    def check(self, item: torch.Tensor, name: str) -> bool:
        """Check if a signal meets the dataset criteria."""
        assert self._fs is not None
        if self.max_length is not None and item.shape[-1] > int(self.max_length * self._fs):
            logger.info(f"'{name}' is longer than `max_length`. Skipping.")
            return False
        if item.pow(2).mean() == 0.0:
            logger.info(f"'{name}' is silent. Skipping.")
            return False
        if self.max_dynamic_range is not None and dynamic_range(item) > self.max_dynamic_range:
            logger.info(f"'{name}' has dynamic range greater than `max_dynamic_range`. Skipping.")
            return False
        return True

    def __len__(self) -> int:
        """Get the number of files in the dataset.

        Returns:
            The number of files in the dataset.

        Note:
            If `segment_length` is not `None`, the number of samples yielded by this dataset when iterating over it does
            not match the output of this method.
        """
        return super().__len__()


class DynamicMixingDataset(ld.ParallelStreamingDataset):
    """Dynamic mixing dataset.

    Wraps two [AudioStreamingDataset][addse.data.AudioStreamingDataset] instances, one for speech and one for noise,
    and generates noisy speech samples on-the-fly by mixing the speech and noise samples at a random signal-to-noise
    ratio (SNR).

    Multi-channel speech and noise samples are converted to mono by randomly selecting one channel.

    If the speech and noise samples have different lengths, the noise is cycled or trimmed to match the speech length.

    When `length=float("inf")`, this dataset is infinite and should be used with `limit_<stage>_batches` in the
    Lightning Trainer.
    """

    def __init__(
        self,
        speech_dataset: AudioStreamingDataset,
        noise_dataset: AudioStreamingDataset,
        snr_range: tuple[float, float] = (-5.0, 15.0),
        rms_range: tuple[float, float] | None = (0.0, 0.0),
        length: int | float | None = float("inf"),
        resume: bool = True,
        reset_rngs: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the dynamic mixing dataset.

        Args:
            speech_dataset: Speech dataset.
            noise_dataset: Noise dataset.
            snr_range: SNR range.
            rms_range: RMS range for the clean speech in dB. If `None`, no RMS adjustment is performed.
            length: Number of samples to yield per epoch. If `None`, the speech and noise datasets are iterated over
                until one is exhausted. If an integer, the datasets are cycled until `length` samples are yielded. If
                `float("inf")`, the datasets are cycled indefinitely.
            resume: Whether to resume the dataset from where it left off in the previous epoch when starting a new
                epoch. Should be set to `False` for validation and test datasets. Only works when iterating with an
                [AudioStreamingDataLoader][addse.data.AudioStreamingDataLoader]. Ignored if `length` is `None`.
            reset_rngs: Whether to set the internal random number generators to the same initial state at the start of
                each epoch. If `True`, random numbers are consistent across epochs. Should be set to `True` for
                validation and test datasets.
            **kwargs: Additional keyword arguments passed to parent constructor.
        """
        if not (isinstance(speech_dataset, AudioStreamingDataset) and isinstance(noise_dataset, AudioStreamingDataset)):
            raise TypeError("`speech_dataset` and `noise_dataset` must be instances of `AudioStreamingDataset`.")
        super().__init__(
            [speech_dataset, noise_dataset],
            transform=functools.partial(self.transform, snr_range=snr_range, rms_range=rms_range),
            length=length,
            reset_rngs=reset_rngs,
            resume=resume,
            **kwargs,
        )

    def __len__(self) -> int:
        """Get the number of samples yielded per epoch.

        Returns:
            Number of samples yielded per epoch.

        Raises:
            TypeError: If the dataset is infinite, i.e. if `length` is `float("inf")`.
        """
        output = super().__len__()
        if output is None:
            raise TypeError(f"{type(self).__name__} has no `len()`.")
        return output

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        """Iterate over the dataset.

        Yields:
            Noisy speech, clean speech, and sample rate. Noisy and clean speech have shape `(1, num_samples)`.
        """
        # When segmenting, AudioStreamingDataset yields a different number of samples than the actual number of files in
        # the dataset, which messes up the internal sample count and breaks StreamingDataLoader and StreamingDataset
        # features. The workaround is to yield the number of files actually loaded in AudioStreamingDataset and adjust
        # the internal sample count accordingly. E.g. if the current sample was obtained by popping the queue and no
        # file was loaded, then fix the internal sample count by subtracting one. We also hard-set iterator._count to 0
        # to always yield `length` samples. See https://github.com/Lightning-AI/litData/issues/642.
        iterator = super().__iter__()
        iterator._count = 0
        for samples in iterator:
            if isinstance(samples, dict) and __NUM_SAMPLES_YIELDED_KEY__ in samples:
                samples[__NUM_SAMPLES_YIELDED_KEY__][0] += samples[__SAMPLES_KEY__][-1][0] - 1
                samples[__NUM_SAMPLES_YIELDED_KEY__][1] += samples[__SAMPLES_KEY__][-1][1] - 1
                assert samples[__NUM_SAMPLES_YIELDED_KEY__][0] > 0
                assert samples[__NUM_SAMPLES_YIELDED_KEY__][0] > 0
                samples[__SAMPLES_KEY__] = samples[__SAMPLES_KEY__][:-1]  # remove the number of files loaded
            else:
                samples = samples[:-1]  # remove the number of files loaded
            yield samples

    @staticmethod
    def transform(
        samples: tuple[ASDOutput, ASDOutput],
        rngs: dict[str, Any],
        snr_range: tuple[float, float],
        rms_range: tuple[float, float] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, int]]:
        """Generate noisy speech from speech and noise samples.

        Args:
            samples: Tuple with speech and noise samples.
            rngs: Random number generators.
            snr_range: SNR range.
            rms_range: RMS range for the clean speech in dB. If `None`, no RMS adjustment is performed.

        Returns:
            Noisy speech, clean speech, sample rate, and number of files loaded. Noisy and clean speech have shape
            `(1, num_samples)`. The number of files loaded is for internal use only and is discarded before yielding
            when iterating over the dataset.
        """
        (speech, speech_fs, _, speech_files_loaded), (noise, noise_fs, _, noise_files_loaded) = samples
        speech = rngs["random"].choice(speech).unsqueeze(0)
        noise = rngs["random"].choice(noise).unsqueeze(0)
        assert speech_fs == noise_fs, "Speech and noise sample rates must be the same."
        if speech.shape != noise.shape:
            # cycle or trim noise to match speech length
            noise = noise[:, torch.arange(speech.shape[-1]) % noise.shape[-1]]
        snr = rngs["random"].uniform(*snr_range)
        if rms_range is not None:
            rms_db = rngs["random"].uniform(*rms_range)
            rms = speech.pow(2).mean().sqrt()
            factor = (10 ** (rms_db / 20)) / rms
            if factor.isfinite():
                speech = speech * factor
            else:
                warnings.warn("Overflow when setting RMS. Returning speech as is.")
        noise = set_snr(speech, noise, snr)
        return speech + noise, speech, speech_fs, (speech_files_loaded, noise_files_loaded)


class AudioStreamingDataLoader(ld.StreamingDataLoader):
    """Audio streaming dataloader."""

    def __init__(
        self,
        dataset: AudioStreamingDataset | DynamicMixingDataset,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the audio streaming dataloader.

        Args:
            dataset: Dataset to wrap.
            batch_size: Batch size.
            num_workers: Number of workers.
            shuffle: Whether to shuffle the dataset at every epoch. If `None`, uses the dataset shuffle attribute.
            **kwargs: Additional keyword arguments passed to parent constructor.
        """
        if not isinstance(dataset, AudioStreamingDataset | DynamicMixingDataset):
            raise TypeError("`dataset` must be an instance of `AudioStreamingDataset` or `DynamicMixingDataset`.")
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, **kwargs)

    def __len__(self) -> int:
        """Get the number of batches in the dataloader.

        Returns:
            Number of batches in the dataloader.

        Raises:
            TypeError: If the wrapped dataset is an instance of
                [AudioStreamingDataset][addse.data.AudioStreamingDataset] with `segment_length!=None`, as the total
                number of segments in the dataset cannot be determined without iterating over it.
        """
        if isinstance(self.dataset, AudioStreamingDataset) and self.dataset.segment_length is not None:
            raise TypeError(f"{type(self).__name__} has no `len()` when `segment_length` is not `None`.")
        return super().__len__()

    @property
    def shuffle(self) -> bool:
        """Get the shuffle attribute of the dataset."""
        if isinstance(self.dataset, AudioStreamingDataset):
            return self.dataset.shuffle
        if isinstance(self.dataset, DynamicMixingDataset):
            assert self.dataset._datasets, "`DynamicMixingDataset` must wrap at least one dataset."
            if not (
                all(ds.shuffle for ds in self.dataset._datasets) or all(not ds.shuffle for ds in self.dataset._datasets)
            ):
                raise ValueError(
                    "All datasets wrapped by `DynamicMixingDataset` must have the same `shuffle` attribute value. "
                    f"Got {[ds.shuffle for ds in self.dataset._datasets]}."
                )
            return self.dataset._datasets[0].shuffle
        raise TypeError(f"Invalid `dataset` type: {type(self.dataset).__name__}.")
