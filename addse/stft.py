import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    """Short-time Fourier transform (STFT) module."""

    window: torch.Tensor

    def __init__(
        self,
        frame_length: int = 512,
        hop_length: int | None = None,
        n_fft: int | None = None,
        window: str = "hann",
        norm: bool = False,
    ) -> None:
        """Initialize the STFT module.

        Args:
            frame_length: Frame length.
            hop_length: Hop length. If `None`, defaults to `frame_length // 2`.
            n_fft: FFT size. If `None`, defaults to `frame_length`.
            window: Window type. Passed to [scipy.signal.get_window][].
            norm: Whether to normalize the window by the square root of its sum of squares.
        """
        super().__init__()

        window_ = scipy.signal.get_window(window, frame_length)
        window_ = torch.from_numpy(window_).float()
        if norm:
            window_ = window_ / window_.pow(2).sum().sqrt()

        self.frame_length = frame_length
        self.hop_length = frame_length // 2 if hop_length is None else hop_length
        self.n_fft = frame_length if n_fft is None else n_fft

        self.register_buffer("window", window_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the STFT.

        Args:
            x: Input tensor. Shape `(..., num_samples)`.

        Returns:
            STFT of input tensor. Shape `(..., num_freqs, num_frames)`.
        """
        assert x.ndim >= 1, f"{type(self).__name__} input must be at least 1-dimensional, got shape {x.shape}."

        # flatten all but last axis
        h = x.reshape(-1, x.shape[-1])

        # pad right to fit last samples
        recon_padding = self.frame_length - self.hop_length
        trail_padding = (self.frame_length - h.shape[-1] - 2 * recon_padding) % self.hop_length
        h = F.pad(h, (0, trail_padding))

        # pad left and right for perfect reconstruction
        h = F.pad(h, (recon_padding, recon_padding))

        # frame and apply window
        h = h.unfold(-1, self.frame_length, self.hop_length)
        h = h * self.window

        # apply fft
        h = torch.fft.rfft(h, dim=-1, n=self.n_fft)

        # swap time and frequency axes
        h = h.swapaxes(-1, -2)

        # restore input shape
        return h.reshape(*x.shape[:-1], h.shape[-2], h.shape[-1])

    def inverse(self, x: torch.Tensor, n: int | None = None) -> torch.Tensor:
        """Compute the inverse STFT.

        Args:
            x: Input tensor. Shape `(..., num_freqs, num_frames)`.
            n: If provided, the output tensor is trimmed to this length along the last axis.

        Returns:
            Reconstructed tensor. Shape `(..., num_samples)`.
        """
        assert x.ndim >= 2, f"{type(self).__name__} inverse input be at least 2-dimensional, got shape {x.shape}."

        # flatten all but frequency and time axis
        h = x.reshape(-1, x.shape[-2], x.shape[-1])

        # apply ifft and trim to frame length
        h = torch.fft.irfft(h, dim=-2)
        h = h[:, : self.frame_length, :]

        # apply window and overlap-add
        h = self.overlap_add(h * self.window.reshape(-1, 1))

        # compute window envelope
        envelope = self.overlap_add(self.window.pow(2).expand(1, h.shape[-1], -1).swapaxes(1, 2))

        # undo perfect reconstruction padding
        padding = self.frame_length - self.hop_length
        envelope = envelope[:, padding : h.shape[-1] - padding]
        h = h[:, padding : h.shape[-1] - padding]

        # normalize by window envelope
        assert envelope.abs().min() > 1e-10, "NOLA constraint is not fulfilled."
        h = h / envelope

        # trim to desired length
        h = h if n is None else h[..., :n]

        # restore input shape
        return h.reshape(*x.shape[:-2], h.shape[-1])

    def overlap_add(self, x: torch.Tensor) -> torch.Tensor:
        """Overlap-add.

        Args:
            x: Input tensor. Shape `(batch_size, num_freqs, num_frames)`.

        Returns:
            Output tensor. Shape `(batch_size, num_samples)`.
        """
        output_size = (x.shape[-1] - 1) * self.hop_length + self.frame_length
        output = F.fold(x, (1, output_size), (1, self.frame_length), stride=(1, self.hop_length))
        return output.squeeze(-2).squeeze(-2)
