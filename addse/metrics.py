import tempfile
import warnings
from abc import abstractmethod
from typing import override

import Levenshtein
import numpy as np
import soundfile as sf
import soxr
import torch
import utmosv2
from discrete_speech_metrics import SpeechBERTScore as SBS
from mel_cepstral_distance import compare_audio_arrays as mel_cepstral_distance
from pesq import BufferTooShortError, NoUtterancesError, pesq
from pystoi import stoi
from torchmetrics.functional.audio import (
    deep_noise_suppression_mean_opinion_score,
    non_intrusive_speech_quality_assessment,
)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class BaseMetric:
    """Base class for metrics."""

    @abstractmethod
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the metric.

        This method should not be called directly. Use `__call__` instead.
        """

    @torch.no_grad()
    def __call__(self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> float:
        """Compute the metric.

        Validates inputs and calls `compute`.

        Args:
            x: Input signal to evaluate. Shape `(num_channels, num_samples)`.
            y: Reference signal to compare against. Shape `(num_channels, num_samples)`.

        Returns:
            Metric value.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError(
                f"Inputs must be numpy arrays or torch tensors. Got {type(x).__name__} and {type(y).__name__}."
            )
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Inputs must be 2-dimensional. Got shapes {x.shape} and {y.shape}.")
        if x.shape != y.shape:
            raise ValueError(f"Inputs must have the same shape. Got {x.shape} and {y.shape}.")
        output = self.compute(x, y)
        if not isinstance(output, float):
            raise TypeError(f"`compute` must return a float. Got {type(output).__name__}.")
        return output


class SDRMetric(BaseMetric):
    """Signal-to-distortion ratio (SDR) metric."""

    def __init__(self, scale_invariant: bool = False, zero_mean: bool = False, eps: float = 1e-7) -> None:
        """Initialize the SDR metric.

        Args:
            scale_invariant: If `True`, computes the scale-invariant signal-to-distortion ratio (SI-SDR).
            zero_mean: If `True`, subtracts the mean from the inputs before computing the metric.
            eps: Small value for numerical stability.
        """
        self.scale_invariant = scale_invariant
        self.zero_mean = zero_mean
        self.eps = eps

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.zero_mean:
            x = x - x.mean()
            y = y - y.mean()
        if self.scale_invariant:
            y = y * (x * y).sum() / (y**2).sum().clip(min=self.eps)
        num = (y**2).sum().clip(min=self.eps)
        den = ((x - y) ** 2).sum().clip(min=self.eps)
        return 10 * np.log10(num / den).item()


class STOIMetric(BaseMetric):
    """Short-time objective intelligibility (STOI) metric.

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(self, fs: int, extended: bool = False) -> None:
        """Initialize the STOI metric.

        Args:
            fs: Sampling frequency of input signals.
            extended: If `True`, computes the extended version of the STOI metric (ESTOI).
        """
        self.fs = fs
        self.extended = extended

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.mean([stoi(y_i, x_i, self.fs, self.extended) for x_i, y_i in zip(x, y)]).item()


class PESQMetric(BaseMetric):
    """Perceptual evaluation of speech quality (PESQ) metric.

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(self, fs: int) -> None:
        """Initialize the PESQ metric.

        Args:
            fs: Sampling frequency of input signals.
        """
        self.fs = fs

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.fs != 16000:
            x = soxr.resample(x.T, self.fs, 16000).T
            y = soxr.resample(y.T, self.fs, 16000).T
        outputs: list[float] = []
        for x_i, y_i in zip(x, y):
            try:
                output = pesq(16000, y_i, x_i, "wb")
            except (BufferTooShortError, NoUtterancesError) as e:
                warnings.warn(f"Error computing PESQ: {e}. Returning NaN.")
                output = float("nan")
            outputs.append(output)
        return np.mean(outputs).item()


class MCDMetric(BaseMetric):
    """Mel-cepstral distance (MCD) metric.

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(self, fs: int) -> None:
        """Initialize the MCD metric."""
        self.fs = fs

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        mcds = [mel_cepstral_distance(x_i, y_i, self.fs, self.fs)[0] for x_i, y_i in zip(x, y)]
        return sum(mcds) / len(mcds)


class DNSMOSMetric(BaseMetric):
    """Deep noise suppression mean opinion score (DNSMOS) metric.

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(self, fs: int) -> None:
        """Initialize the DNS-MOS metric.

        Args:
            fs: Sampling frequency of input signals.
        """
        self.fs = fs

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        scores = deep_noise_suppression_mean_opinion_score(torch.from_numpy(x), self.fs, False)
        return scores[..., 0].mean().item()


class NISQAMetric(BaseMetric):
    """Non-intrusive speech quality assessment (NISQA) metric.

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(self, fs: int) -> None:
        """Initialize the NISQA metric."""
        self.fs = fs

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        scores = non_intrusive_speech_quality_assessment(torch.from_numpy(x), self.fs)
        return scores.mean().item()


class UTMOSMetric(BaseMetric):
    """UTokyo-SaruLab MOS prediction system (UTMOSv2).

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(self, fs: int, device: str = "auto") -> None:
        """Initialize the PESQ metric.

        Args:
            fs: Sampling frequency of input signals.
            device: Device to run the model on. One of 'auto', 'cpu', or 'cuda'.
        """
        assert device in ["auto", "cpu", "cuda"], f"Device must be 'auto', 'cpu', or 'cuda'. Got {device}."
        self.fs = fs
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = utmosv2.create_model(pretrained=True, device=self.device)

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        with tempfile.TemporaryDirectory() as tempdir:
            for i, x_i in enumerate(x):
                peak_i = np.abs(x_i).max()
                sf.write(f"{tempdir}/input_{i}.wav", x_i / peak_i, self.fs)
            preds = self.model.predict(input_dir=tempdir, device=self.device)
            return sum(pred["predicted_mos"] for pred in preds) / len(preds)


class SCOREQMetric(BaseMetric):
    """Speech contrastive regression for quality assessment (SCOREQ).

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(self, fs: int) -> None:
        """Initialize the SCOREQ metric."""
        # import here in __init__ instead of at the top
        # see https://github.com/alessandroragano/scoreq/pull/4
        from scoreq.scoreq import Scoreq, dynamic_pad

        self.fs = fs
        self.scoreq = Scoreq(data_domain="natural", mode="nr", use_onnx=True)
        self.dynamic_pad = dynamic_pad

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        input_name = self.scoreq.session.get_inputs()[0].name
        if self.fs != 16000:
            x = soxr.resample(x.T, self.fs, 16000).T
        x = self.dynamic_pad(torch.from_numpy(x).float()).numpy()
        (output,) = self.scoreq.session.run(None, {input_name: x})
        return output.mean().item()


class LPSMetric(BaseMetric):
    """Levenshtein phoneme similarity (LPS).

    Calculated independently for each channel and averaged across channels.
    """

    def __init__(
        self,
        fs: int,
        device: str = "auto",
        checkpoint: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
    ) -> None:
        """Initialize the LPS metric."""
        assert device in ["auto", "cpu", "cuda"], f"Device must be 'auto', 'cpu', or 'cuda'. Got {device}."
        self.fs = fs
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to(device=self.device)

    def _phoneme_predictor(self, x: np.ndarray) -> list[str]:
        input_values = self.processor(x, return_tensors="pt", sampling_rate=16000).input_values
        logits = self.model(input_values.to(device=self.device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.fs != 16000:
            x = soxr.resample(x.T, self.fs, 16000).T
            y = soxr.resample(y.T, self.fs, 16000).T
        x_phonemes = self._phoneme_predictor(x)
        y_phonemes = self._phoneme_predictor(y)
        lps = [self._lps(x, y) for x, y in zip(x_phonemes, y_phonemes)]
        return sum(lps) / len(lps)

    def _lps(self, x: str, y: str) -> float:
        x, y = x.replace(" ", ""), y.replace(" ", "")
        if len(y) == 0:
            return float("nan")
        lev_distance = Levenshtein.distance(x, y)
        return 1 - lev_distance / len(y)


class SBSMetric(BaseMetric):
    """SpeechBERTScore (SBS)."""

    def __init__(self, fs: int, device: str = "auto") -> None:
        """Initialize the SBS metric."""
        assert device in ["auto", "cpu", "cuda"], f"Device must be 'auto', 'cpu', or 'cuda'. Got {device}."
        self.fs = fs
        self.sbs = SBS(sr=16000, model_type="hubert-base", layer=8, use_gpu=device in ["auto", "cuda"])

    @override
    def compute(self, x: np.ndarray, y: np.ndarray) -> float:
        if self.fs != 16000:
            x = soxr.resample(x.T, self.fs, 16000).T
            y = soxr.resample(y.T, self.fs, 16000).T
        precisions = [self.sbs.score(y_i, x_i)[0] for x_i, y_i in zip(x, y)]
        return sum(precisions) / len(precisions)


# metrics supporting multiprocessing
MP_METRICS = (SDRMetric, STOIMetric, PESQMetric, MCDMetric)
