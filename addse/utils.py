import io
import itertools
import math
import os
import random
import re
import warnings
from collections.abc import Iterable, Iterator
from typing import Any, Literal

import lightning as L
import numpy as np
import soundfile as sf
import soxr
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig


def dynamic_range(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Dynamic range in dB.

    Calculated as the ratio between the peak amplitude and the RMS.

    Args:
        x: Input signal. Any number of dimensions.
        eps: Small value for numerical stability.

    Returns:
        Dynamic range in dB.
    """
    peak = x.abs().max().clamp(min=eps)
    rms = x.pow(2).mean().sqrt().clamp(min=eps)
    return 20 * torch.log10(peak / rms)


def set_snr(speech: torch.Tensor, noise: torch.Tensor, snr: float) -> torch.Tensor:
    """Scale noise to achieve a desired signal-to-noise ratio (SNR).

    Args:
        speech: Speech signal. Any number of dimensions.
        noise: Noise signal. Any number of dimensions.
        snr: Desired SNR in dB.

    Returns:
        Scaled noise signal.
    """
    assert speech.shape == noise.shape, f"Inputs must have same shape, got {speech.shape} and {noise.shape}."
    num = speech.pow(2).sum()
    den = noise.pow(2).sum()
    factor = 10 ** (-snr / 20) * (num / den) ** 0.5
    if factor.isfinite():
        noise = noise * factor
    else:
        warnings.warn("Overflow when setting SNR. Returning noise as is.")
    return noise


def seed_all(seed: int) -> None:
    """Set the seed for all random number generators.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def segment_audio_file(
    path: str,
    format: str = "ogg",
    subtype: str | None = None,
    seglen: float | None = None,
    base: str | None = None,
) -> Iterator[tuple[bytes, str]]:
    """Read and segment an audio file and yield bytes and a name for each segment.

    Args:
        path: Path to the input audio file.
        format: Audio format to convert to.
            See [soundfile.write](https://python-soundfile.readthedocs.io/en/latest/#soundfile.write).
        subtype: Audio subtype to convert to.
            See [soundfile.write](https://python-soundfile.readthedocs.io/en/latest/#soundfile.write).
        seglen: Segment length in seconds. If provided, the file is segmented into chunks of this length approximately.
        base: Base path to strip from the file path.

    Yields:
        Audio bytes and name.
    """
    relpath_noext, _ = os.path.splitext(os.path.relpath(path, base))
    x, fs = sf.read(path, always_2d=True)
    opus_fs = [8000, 12000, 16000, 24000, 48000]
    if format == "ogg" and subtype == "opus" and fs not in opus_fs:
        if fs > opus_fs[-1]:
            raise ValueError(f"Opus only supports sample rates of 8000, 12000, 16000, 24000, and 48000. Got {fs}.")
        next_fs = next(f for f in opus_fs if f > fs)
        warnings.warn(
            f"Opus only supports sample rates of 8000, 12000, 16000, 24000, and 48000. Got {fs}. "
            f"Resampling to {next_fs}.",
        )
        x = soxr.resample(x, fs, next_fs)
        fs = next_fs
    lenx = len(x)
    seglen = lenx if seglen is None else int(seglen * fs)
    nseg = max(1, round(lenx / seglen))
    x = x[: lenx - lenx % nseg]
    x = x.reshape((nseg, -1, x.shape[-1]))
    digits = math.ceil(math.log10(nseg))
    for i_seg, segment in enumerate(x):
        buffer = io.BytesIO()
        sf.write(buffer, segment, fs, format=format, subtype=subtype)
        buffer.seek(0)
        name = f"{relpath_noext}{'' if seglen is None else f'_{i_seg:0{digits}d}'}.{format}"
        yield buffer.read(), name


def scan_files(input_dir: str, regex: str) -> Iterator[str]:
    """Scan a directory for files matching a regular expression.

    Args:
        input_dir: Directory to scan.
        regex: Regular expression to match file paths.

    Yields:
        Path matching the regular expression.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            path = os.path.join(root, file)
            if re.match(regex, path):
                yield path


def bytes_str_to_int(bytes_str: str) -> int:
    """Convert a human-readable byte size to an integer.

    Args:
        bytes_str: Human-readable byte size (e.g., "64MB", "1GB").

    Returns:
        Integer byte size.
    """
    suffixes = {
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
    }
    for suffix, size in suffixes.items():
        if bytes_str.endswith(suffix):
            try:
                return int(bytes_str[: -len(suffix)]) * size
            except ValueError:
                raise ValueError(f"Invalid size: {bytes_str}")
    raise ValueError(f"Invalid size: {bytes_str}")


def build_subbands(n_fft: int, fs: int, subbands: Iterable[tuple[float, int]]) -> list[tuple[int, int]]:
    """Derive subband indices on the FFT axis.

    Args:
        n_fft: FFT size.
        fs: Sampling rate.
        subbands: List of tuples `(bandwidth, number)`, where `bandwidth` is the bandwidth of the subband in Hz
            and `number` is the number of subbands with that bandwidth.

    Returns:
        List of tuples `(start, end)`, where `start` and `end` are the start and end indices of the subband on
        the FFT axis.
    """
    bandwidths = sum(([bandwidth] * number for bandwidth, number in subbands), [])
    right_limits = list(itertools.accumulate(bandwidths))
    df = fs / n_fft
    rfftfreqs = torch.arange(n_fft // 2 + 1) * df
    right_limits_idx = [
        int(torch.where(rfftfreqs > right_limit)[0][0]) for right_limit in right_limits if right_limit < rfftfreqs[-1]
    ]
    # add last subband
    last_right_limit_idx = min(n_fft // 2 + 1, int(right_limits[-1] // df) + 1)
    if right_limits_idx[-1] < last_right_limit_idx:
        right_limits_idx.append(last_right_limit_idx)
    assert len(right_limits) == len(right_limits_idx), "Got subbands above Nyquist frequency."
    assert len(set(right_limits_idx)) == len(right_limits_idx), "Got empty subbands. Increase bandwidth or n_fft."
    return [(0 if i == 0 else right_limits_idx[i - 1], right_limits_idx[i]) for i in range(len(right_limits_idx))]


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten.
        parent_key: Key prefix for the current level.
        sep: Separator to use between keys.

    Returns:
        Flattened dictionary.
    """
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Unflatten a dictionary.

    Args:
        d: Dictionary to unflatten.
        sep: Separator used between keys.

    Returns:
        Unflattened dictionary.
    """
    result_dict: dict[str, Any] = {}
    for key, value in d.items():
        parts = key.split(sep)
        current_dict = result_dict
        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        current_dict[parts[-1]] = value
    return result_dict


def hz_to_mel(hz: float, scale: str = "slaney") -> float:
    """Convert frequency in Hz to mel scale.

    Args:
        hz: Frequency in Hz.
        scale: Mel scale to use. `"htk"` matches the Hidden Markov Toolkit, while `"slaney"` matches the Auditory
            Toolbox by Slaney. The `"slaney"` scale is linear below 1 kHz and logarithmic above 1 kHz.

    Returns:
        Frequency in mel scale.
    """
    if scale == "htk":
        return 2595 * math.log10(1 + hz / 700)
    if scale == "slaney":
        break_freq = 1000  # limit between linear and logarithmic regions
        slope = 3 / 200  # slope in linear region
        return slope * hz if hz < break_freq else slope * break_freq * (1 + math.log(hz / break_freq))
    raise ValueError(f"Invalid mel scale: {scale}")


def mel_to_hz(mel: torch.Tensor, scale: str = "slaney") -> torch.Tensor:
    """Convert frequency in mel scale to Hz.

    Args:
        mel: Frequency in mel scale.
        scale: Mel scale to use. `"htk"` matches the Hidden Markov Toolkit, while `"slaney"` matches the Auditory
            Toolbox by Slaney. The `"slaney"` scale is linear below 1 kHz and logarithmic above 1 kHz.

    Returns:
        Frequency in Hz.
    """
    if scale == "htk":
        return 700 * (10 ** (mel / 2595) - 1)
    if scale == "slaney":
        break_freq = 1000  # limit between linear and logarithmic regions
        slope = 3 / 200  # slope in linear region
        break_mel = slope * break_freq
        hz = mel / slope
        mask = mel >= break_mel
        hz[mask] = break_freq * torch.exp(mel[mask] / break_mel - 1)
        return hz
    raise ValueError(f"Invalid mel scale: {scale}")


def mel_filters(
    n_filters: int = 64,
    n_fft: int = 512,
    f_min: float = 0.0,
    f_max: float | None = None,
    fs: float = 16000,
    scale: str = "slaney",
    norm: Literal["slaney", "consistent"] | None = "consistent",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get mel filters.

    Args:
        n_filters: Number of filters.
        n_fft: Number of FFT point.
        f_min: Minimum frequency.
        f_max: Maximum frequency. If `None`, uses `fs / 2`.
        fs: Sampling frequency.
        scale: Mel scale to use. `"htk"` matches the Hidden Markov Toolkit, while `"slaney"` matches the Auditory
            Toolbox by Slaney. The `"slaney"` scale is linear below 1 kHz and logarithmic above 1 kHz.
        norm: Filter normalization method. If `"slaney"`, the filters are normalized by their width in Hz. However this
            makes the filter response scale with the frequency resolution `n_fft / fs`. If `"consistent"`, the frequency
            resolution is factored in. If `None`, no normalization is applied.
        dtype: Data type to cast the filters to.

    Returns:
        Mel filters and center frequencies. Shapes `(n_filters, n_fft // 2 + 1)` and `(n_filters,)`.
    """
    if f_max is not None and f_max > fs / 2:
        raise ValueError(f"f_max ({f_max}) must be less than or equal to fs / 2 ({fs / 2}).")
    f_max = fs / 2 if f_max is None else f_max
    mel_min = hz_to_mel(f_min, scale)
    mel_max = hz_to_mel(f_max, scale)
    mel = torch.linspace(mel_min, mel_max, n_filters + 2, dtype=dtype)
    fc = mel_to_hz(mel, scale)
    f = torch.arange(n_fft // 2 + 1, dtype=dtype) * fs / n_fft
    dfc = fc.diff().unsqueeze(1)
    slopes = fc.unsqueeze(1) - f.unsqueeze(0)
    down_slopes = -slopes[:-2] / dfc[:-1]
    up_slopes = slopes[2:] / dfc[1:]
    filters = torch.min(down_slopes, up_slopes).clamp(min=0)
    if any(filters.sum(dim=1) == 0):
        raise ValueError("Got empty mel filters. Increase f_max or n_fft, or decrease n_filters.")
    if norm == "slaney":
        filters /= 0.5 * (fc[2:] - fc[:-2]).unsqueeze(1)
    elif norm == "consistent":
        filters /= 0.5 * n_fft / fs * (fc[2:] - fc[:-2]).unsqueeze(1)
    elif norm is not None:
        raise ValueError(f"Invalid norm: {norm}")
    return filters, fc


def load_hydra_config(path: str, overrides: list[str] | None = None) -> tuple[DictConfig, str]:
    """Load a Hydra configuration file."""
    config_dir, config_name = os.path.split(os.path.abspath(path))
    config_name, _ = os.path.splitext(config_name)
    initialize_config_dir(config_dir=config_dir, version_base=None)
    return compose(config_name=config_name, overrides=overrides or []), config_name


def load_model(
    config_path: str,
    model_name: str | None = None,
    logs_dir: str = "logs",
    ckpt_name: str = "last.ckpt",
    ckpt_path: str | None = None,
    state_key: str | None = "state_dict",
    prepend_key: str | None = None,
    device: torch.device | str | None = None,
    strict: bool = True,
) -> L.LightningModule:
    """Load a model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg, cfg_name = load_hydra_config(config_path)
    model_name = model_name or cfg_name
    lm = instantiate(cfg.lm).to(device)
    if ckpt_name is None and ckpt_path is None:
        return lm.eval()
    ckpt_path = ckpt_path or os.path.join(logs_dir, model_name, "checkpoints", ckpt_name)
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = state_dict[state_key] if state_key is not None else state_dict
    state_dict = {f"{prepend_key}.{k}": v for k, v in state_dict.items()} if prepend_key else state_dict
    lm.load_state_dict(state_dict, strict=strict)
    return lm.eval()
