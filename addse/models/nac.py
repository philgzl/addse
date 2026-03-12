import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class NACSnakeActivation(nn.Module):
    """Neural audio codec Snake activation function."""

    def __init__(self, channels: int) -> None:
        """Initialize the neural audio codec Snake activation function."""
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + (self.alpha[:, None] * x).sin() ** 2 / (self.alpha[:, None] + 1e-9)


class NACConv1d(nn.Module):
    """Neural audio codec 1D convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: tuple[int, int] | str = (0, 0),
        dilation: int = 1,
        activation: bool = True,
        bias: bool = True,
    ) -> None:
        """Initialize the neural audio codec 1D convolutional layer."""
        super().__init__()
        self.padding = padding
        self.act = NACSnakeActivation(in_channels) if activation else nn.Identity()
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding if isinstance(padding, str) else 0,
                dilation,
                bias=bias,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.pad(x, self.padding) if isinstance(self.padding, tuple) else x
        return self.conv(self.act(x))


class NACConvTranspose1d(nn.Module):
    """Neural audio codec 1D transposed convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initialize the neural audio codec 1D transposed convolutional layer."""
        super().__init__()
        self.act = NACSnakeActivation(in_channels)
        self.conv = weight_norm(nn.ConvTranspose1d(in_channels, out_channels, 2 * stride, stride))
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(self.act(x))
        return x[..., self.stride // 2 : x.shape[-1] - (self.stride // 2 + self.stride % 2)]


class NACResidualUnit(nn.Module):
    """Neural audio codec residual unit."""

    def __init__(self, channels: int, dilation: int, kernel_size: int) -> None:
        """Initialize the neural audio codec residual unit."""
        super().__init__()
        self.conv1 = NACConv1d(channels, channels, kernel_size, padding="same", dilation=dilation)
        self.conv2 = NACConv1d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x + self.conv2(self.conv1(x))


class NACEncoderBlock(nn.Module):
    """Neural audio codec encoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        num_residual_units: int,
        dilation_base: int,
    ) -> None:
        """Initialize the neural audio codec encoder block."""
        super().__init__()
        self.residual_blocks = nn.Sequential(
            *[NACResidualUnit(in_channels, dilation_base**i, kernel_size) for i in range(num_residual_units)]
        )
        self.conv = NACConv1d(in_channels, out_channels, 2 * stride, stride, (stride // 2 + stride % 2, stride // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(self.residual_blocks(x))


class NACDecoderBlock(nn.Module):
    """Neural audio codec decoder block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        kernel_size: int,
        num_residual_units: int,
        dilation_base: int,
    ) -> None:
        """Initialize the neural audio codec decoder block."""
        super().__init__()
        self.conv = NACConvTranspose1d(in_channels, out_channels, stride)
        self.residual_blocks = nn.Sequential(
            *[NACResidualUnit(out_channels, dilation_base**i, kernel_size) for i in range(num_residual_units)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.residual_blocks(self.conv(x))


class NACLSTMBlock(nn.Module):
    """Neural audio codec LSTM block."""

    def __init__(self, channels: int) -> None:
        """Initialize the neural audio codec LSTM block."""
        super().__init__()
        assert channels % 2 == 0, "channels must be even for bidirectional LSTM"
        self.lstm = nn.LSTM(channels, channels // 2, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = x.transpose(1, 2)
        h, _ = self.process_in_blocks(h)
        return x + h.transpose(1, 2)

    def process_in_blocks(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input in blocks to avoid LSTM length limitation.

        See https://github.com/pytorch/pytorch/issues/133751.
        """
        BLOCK_SIZE = 65_535
        if not x.shape[1] > BLOCK_SIZE:
            return self.lstm(x)
        hidden_state: torch.Tensor | None = None
        outputs = []
        for i in range(0, x.shape[1], BLOCK_SIZE):
            block = x[:, i : i + BLOCK_SIZE]
            output, hidden_state = self.lstm(block, hidden_state)
            outputs.append(output)
        assert hidden_state is not None
        return torch.cat(outputs, dim=1), hidden_state


class NACEncoder(nn.Module):
    """Neural audio codec encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        strides: list[int],
        kernel_size: int,
        num_residual_units: int,
        dilation_base: int,
        in_kernel_size: int,
        out_kernel_size: int,
    ) -> None:
        """Initialize the neural audio codec encoder."""
        super().__init__()
        self.in_conv = NACConv1d(in_channels, base_channels, in_kernel_size, padding="same", activation=False)
        self.blocks = nn.Sequential(
            *[
                module
                for i, stride in enumerate(strides)
                for module in [
                    NACEncoderBlock(
                        base_channels * (2**i),
                        base_channels * (2 ** (i + 1)),
                        stride,
                        kernel_size,
                        num_residual_units,
                        dilation_base,
                    ),
                    NACLSTMBlock(base_channels * (2 ** (i + 1))),
                ]
            ]
        )
        self.out_conv = NACConv1d(base_channels * (2 ** len(strides)), out_channels, out_kernel_size, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input audio into continuous embeddings.

        Args:
            x: Input audio. Shape `(batch_size, in_channels, num_samples)`.

        Returns:
            Continuous embeddings. Shape `(batch_size, out_channels, num_frames)`.
        """
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        return self.out_conv(self.blocks(self.in_conv(x)))


class NACDecoder(nn.Module):
    """Neural audio codec decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        strides: list[int],
        kernel_size: int,
        num_residual_units: int,
        dilation_base: int,
        in_kernel_size: int,
        out_kernel_size: int,
    ) -> None:
        """Initialize the neural audio codec decoder."""
        super().__init__()
        self.in_conv = NACConv1d(
            in_channels, base_channels * (2 ** len(strides)), in_kernel_size, padding="same", activation=False
        )
        self.blocks = nn.Sequential(
            *[
                NACDecoderBlock(
                    base_channels * (2 ** (i + 1)),
                    base_channels * (2**i),
                    stride,
                    kernel_size,
                    num_residual_units,
                    dilation_base,
                )
                for i, stride in reversed(list(enumerate(strides)))
            ]
        )
        self.out_conv = NACConv1d(base_channels, out_channels, out_kernel_size, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode continuous embeddings into audio.

        Args:
            x: Continuous embeddings. Shape `(batch_size, in_channels, num_frames)`.

        Returns:
            Decoded audio. Shape `(batch_size, out_channels, num_samples)`.
        """
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        return self.out_conv(self.blocks(self.in_conv(x)))


class NACVQVAE(nn.Module):
    """Neural audio codec vector quantizer."""

    def __init__(
        self,
        emb_channels: int,
        codebook_size: int,
        codebook_channels: int | None,
        normalize: bool,
        codebook: nn.Embedding | None,
    ) -> None:
        """Initialize the neural audio codec vector quantizer."""
        super().__init__()
        self.in_conv = (
            nn.Identity()
            if codebook_channels is None
            else NACConv1d(emb_channels, codebook_channels, activation=False, bias=False)
        )
        self.out_conv = (
            nn.Identity()
            if codebook_channels is None
            else NACConv1d(codebook_channels, emb_channels, activation=False, bias=False)
        )
        self.codebook = (
            nn.Embedding(codebook_size, emb_channels if codebook_channels is None else codebook_channels)
            if codebook is None
            else codebook
        )
        self.normalize = normalize

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign discrete codes to continuous input embeddings.

        Args:
            x: Input continuous embeddings. Shape `(batch_size, emb_channels, num_frames)`

        Returns:
            A tuple `(codes, quantized, codebook_loss, commit_loss, x_proj, quantized_proj)`:

            - `codes`: Assigned vector indices with shape `(batch_size, num_frames)`.
            - `quantized`: Quantized embeddings with shape `(batch_size, emb_channels, num_frames)`.
            - `codebook_loss`: Codebook loss. 0-dimensional.
            - `commit_loss`: Commitment loss. 0-dimensional.
            - `x_proj`: Projected input embeddings. Shape `(batch_size, codebook_channels, num_frames)`.
            - `quantized_proj`: Projected quantized embeddings. Shape `(batch_size, codebook_channels, num_frames)`.
        """
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        x_proj = self.in_conv(x)
        codes, quantized_proj, codebook_loss, commit_loss = self.quantize(x_proj)
        quantized = self.out_conv(quantized_proj)
        return codes, quantized, codebook_loss, commit_loss, x_proj, quantized_proj

    def quantize(self, x_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize projected input embeddings."""
        embeddings = F.normalize(x_proj, dim=1) if self.normalize else x_proj
        codebook = F.normalize(self.codebook.weight, dim=1) if self.normalize else self.codebook.weight
        dists = torch.cdist(embeddings.transpose(1, 2), codebook.unsqueeze(0))
        codes = torch.argmin(dists, dim=-1)
        quantized = self.codebook(codes).transpose(1, 2)
        codebook_loss = F.mse_loss(x_proj.detach(), quantized)
        commit_loss = F.mse_loss(x_proj, quantized.detach())
        quantized_proj = x_proj + (quantized - x_proj).detach()  # straight-through estimator
        return codes, quantized_proj, codebook_loss, commit_loss

    def decode(self, x: torch.Tensor, domain: str = "code") -> torch.Tensor:
        """Decode input into quantized embeddings.

        Args:
            x: Input tensor:

                - Shape `(batch_size, num_frames)` if `domain` is `"code"`.
                - Shape `(batch_size, emb_channels, num_frames)` if `domain` is `"x"`.
                - Shape `(batch_size, emb_channels, num_frames)` if `domain` is `"q"`.
                - Shape `(batch_size, codebook_channels, num_frames)` if `domain` is `"x_proj"`.
                - Shape `(batch_size, codebook_channels, num_frames)` if `domain` is `"q_proj"`.
            domain: Domain of input tensor.

        Returns:
            Decoded tensor. Shape `(batch_size, emb_channels, num_frames)`
        """
        if domain == "code":
            assert x.ndim == 2, f"Input must be 2-dimensional, got shape {x.shape}"
            quantized_proj = self.codebook(x).transpose(1, 2)
            return self.out_conv(quantized_proj)
        assert x.ndim == 3, f"Input must be 3-dimensional, got shape {x.shape}"
        if domain == "x":
            _, quantized, _, _, _, _ = self.forward(x)
            return quantized
        if domain == "q":
            return x
        if domain == "x_proj":
            _, quantized_proj, _, _ = self.quantize(x)
            return self.out_conv(quantized_proj)
        if domain == "q_proj":
            return self.out_conv(x)
        raise ValueError(f"Unknown domain: {domain}")


class NACRVQVAE(nn.Module):
    """Neural audio codec residual vector quantizer."""

    def __init__(
        self,
        emb_channels: int,
        codebook_size: int,
        num_codebooks: int,
        codebook_channels: int | None,
        normalize: bool,
        shared_codebook: bool,
    ) -> None:
        """Initialize the neural audio codec residual vector quantizer."""
        super().__init__()
        codebook = (
            nn.Embedding(codebook_size, emb_channels if codebook_channels is None else codebook_channels)
            if shared_codebook
            else None
        )
        self.codebooks = nn.ModuleList(
            [
                NACVQVAE(emb_channels, codebook_size, codebook_channels, normalize, codebook)
                for _ in range(num_codebooks)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        no_sum: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign discrete codes to continuous input embeddings.

        Args:
            x: Input continuous embeddings. Shape `(batch_size, emb_channels, num_frames)`
            no_sum: If `True`, the quantized embeddings are not summed across codebooks.

        Returns:
            A tuple `(codes, quantized, codebook_loss, commit_loss, x_proj, quantized_proj)`:

            - `codes`: Assigned vector indices. Shape `(batch_size, num_codebooks, num_frames)`.
            - `quantized` Quantized embeddings. Shape `(batch_size, emb_channels, num_frames)` if `no_sum` is `False`
                else `(batch_size, emb_channels, num_codebooks, num_frames)`.
            - `codebook_loss`: Codebook loss. 0-dimensional.
            - `commit_loss`: Commitment loss. 0-dimensional.
            - `x_proj`: Projected input embeddings. Shape `(batch_size, codebook_channels, num_codebooks, num_frames)`.
            - `quantized_proj`: Projected quantized embeddings. Shape `(batch_size, codebook_channels, num_codebooks,
                num_frames)`.
        """
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        residual = x
        all_codes = []
        all_quantized = []
        all_x_proj = []
        all_quantized_proj = []
        total_codebook_loss: torch.Tensor | float = 0.0
        total_commit_loss: torch.Tensor | float = 0.0
        for codebook in self.codebooks:
            codes, quantized, codebook_loss, commit_loss, x_proj, quantized_proj = codebook(residual)
            all_codes.append(codes)
            all_quantized.append(quantized)
            all_x_proj.append(x_proj)
            all_quantized_proj.append(quantized_proj)
            total_codebook_loss += codebook_loss
            total_commit_loss += commit_loss
            residual = residual - quantized
        codes = torch.stack(all_codes, dim=1)
        quantized = torch.stack(all_quantized, dim=2) if no_sum else sum(all_quantized)
        x_proj = torch.stack(all_x_proj, dim=2)
        quantized_proj = torch.stack(all_quantized_proj, dim=2)
        assert isinstance(total_codebook_loss, torch.Tensor)
        assert isinstance(total_commit_loss, torch.Tensor)
        return codes, quantized, total_codebook_loss, total_commit_loss, x_proj, quantized_proj

    def decode(
        self, x: torch.Tensor, input_no_sum: bool = False, output_no_sum: bool = False, domain: str = "code"
    ) -> torch.Tensor:
        """Decode input into quantized embeddings.

        Args:
            x: Input tensor:

                - If `domain` is `"code"`: Shape `(batch_size, num_codebooks, num_frames)`.
                - If `domain` is `"x"`: Shape `(batch_size, emb_channels, num_frames)`.
                - If `domain` is `"q"`: Shape `(batch_size, emb_channels, num_frames)` if `input_no_sum` is `False` else
                    `(batch_size, emb_channels, num_codebooks, num_frames)`.
                - If `domain` is `"x_proj"`: Shape `(batch_size, codebook_channels, num_codebooks, num_frames)`.
                - If `domain` is `"q_proj"`: Shape `(batch_size, codebook_channels, num_codebooks, num_frames)`.
            input_no_sum: If `False`, the input quantized embeddings are assumed to be summed across codebooks. Ignored
                if `domain` is not `"q"`.
            output_no_sum: If `True`, the output quantized embeddings are not summed across codebooks.
            domain: Domain of input tensor.

        Returns:
            Decoded tensor. Shape `(batch_size, emb_channels, num_frames)` if `output_no_sum` is `False` else
            `(batch_size, emb_channels, num_codebooks, num_frames)`.
        """
        if domain == "x":
            assert x.ndim == 3, f"Input must be 3-dimensional, got shape {x.shape}"
            _, quantized, _, _, _, _ = self.forward(x, no_sum=output_no_sum)
            return quantized
        if domain == "q":
            assert not input_no_sum or x.ndim == 4, f"Input must be 4-dimensional, got shape {x.shape}"
            assert not input_no_sum or x.shape[-2] == len(self.codebooks), (
                f"Input shape ({x.shape}) must match number of codebooks ({len(self.codebooks)}) along dimension -2."
            )
            assert input_no_sum or x.ndim == 3, f"Input must be 3-dimensional, got shape {x.shape}"
            if (input_no_sum and output_no_sum) or (not input_no_sum and not output_no_sum):
                return x
            if input_no_sum and not output_no_sum:
                return x.sum(dim=2)
            if not input_no_sum and output_no_sum:
                raise ValueError("Cannot convert summed quantized embeddings to non-summed.")
        assert domain != "code" or x.ndim == 3, f"Input must be 3-dimensional, got shape {x.shape}"
        assert domain not in ("x_proj", "q_proj") or x.ndim == 4, f"Input must be 4-dimensional, got shape {x.shape}"
        assert x.shape[-2] == len(self.codebooks), (
            f"Input shape ({x.shape}) must match number of codebooks ({len(self.codebooks)}) along dimension -2."
        )
        quantized_list = [codebook.decode(x[..., i, :], domain=domain) for i, codebook in enumerate(self.codebooks)]  # type: ignore
        return torch.stack(quantized_list, dim=2) if output_no_sum else sum(quantized_list)


class NAC(nn.Module):
    """Neural audio codec."""

    def __init__(
        self,
        in_channels: int = 1,
        emb_channels: int = 1024,
        base_channels: int = 32,
        strides: list[int] = [2, 2, 4, 4, 5],
        kernel_size: int = 3,
        num_residual_units: int = 3,
        dilation_base: int = 3,
        encoder_in_kernel_size: int = 7,
        encoder_out_kernel_size: int = 7,
        decoder_in_kernel_size: int = 7,
        decoder_out_kernel_size: int = 7,
        codebook_channels: int | None = 8,
        codebook_size: int = 1024,
        num_codebooks: int = 4,
        normalize: bool = True,
        shared_codebook: bool = False,
    ) -> None:
        """Initialize the neural audio codec.

        Args:
            in_channels: Number of input channels.
            emb_channels: Number of output and input channels for the encoder and decoder, respectively.
            base_channels: Number of base channels for the encoder and decoder.
            strides: Downsampling and upsampling factors for the encoder and decoder blocks, respectively.
            kernel_size: Kernel size for the residual units.
            num_residual_units: Number of residual units per encoder and decoder block.
            dilation_base: Dilation base for the residual units.
            encoder_in_kernel_size: Kernel size for the encoder input convolutional layer.
            encoder_out_kernel_size: Kernel size for the encoder output convolutional layer.
            decoder_in_kernel_size: Kernel size for the decoder input convolutional layer.
            decoder_out_kernel_size: Kernel size for the decoder output convolutional layer.
            codebook_channels: Number of channels for the codebook vectors. If `None`, uses `emb_channels`. Else, each
                quantizer uses input and output linear layers to map between `emb_channels` and `codebook_channels`.
            codebook_size: Number of vectors per codebook.
            num_codebooks: Number of codebooks.
            normalize: Whether to normalize the embeddings and codebook vectors before codebook lookup.
            shared_codebook: Whether to use the same codebook for all quantizers.
        """
        super().__init__()
        self.encoder = NACEncoder(
            in_channels,
            emb_channels,
            base_channels,
            strides,
            kernel_size,
            num_residual_units,
            dilation_base,
            encoder_in_kernel_size,
            encoder_out_kernel_size,
        )
        self.decoder = NACDecoder(
            emb_channels,
            in_channels,
            base_channels,
            strides,
            kernel_size,
            num_residual_units,
            dilation_base,
            decoder_in_kernel_size,
            decoder_out_kernel_size,
        )
        self.quantizer = NACRVQVAE(
            emb_channels, codebook_size, num_codebooks, codebook_channels, normalize, shared_codebook
        )
        self.downsampling_factor = math.prod(strides)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input audio. Shape `(batch_size, in_channels, num_samples)`.

        Returns:
            Tuple `(decoded, codes, codebook_loss, commit_loss)` where `decoded` is the reconstructed audio with shape
            `(batch_size, in_channels, num_samples)`, `codes` are the discrete codes with shape `(batch_size,
            num_codebooks, num_frames)`, `codebook_loss` is the codebook loss, and `commit_loss` is the commitment loss.
        """
        assert x.ndim == 3, f"{type(self).__name__} input must be 3-dimensional, got shape {x.shape}"
        assert x.shape[-1] % self.downsampling_factor == 0, (
            f"Input size along last dimension must be divisible by {self.downsampling_factor}. Got shape {x.shape}."
        )
        encoded = self.encoder(x)
        codes, quantized, codebook_loss, commit_loss, _, _ = self.quantizer(encoded)
        decoded = self.decoder(quantized)
        return decoded, codes, codebook_loss, commit_loss

    def encode(self, x: torch.Tensor, no_sum: bool = False, domain: str = "q") -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input audio into discrete codes.

        Args:
            x: Input audio. Shape `(batch_size, in_channels, num_samples)`.
            no_sum: If `True`, the quantized embeddings are not summed across codebooks. Ignored if `domain` is not
                `"q"`.
            domain: Which continuous output to return. One of:

                - `"x"`: Return the encoder output.
                - `"q"`: Return the quantized embeddings.
                - `"x_proj"`: Return the projected encoder output in codebook space.
                - `"q_proj"`: Return the projected quantized embeddings in codebook space.

        Returns:
            Tuple `(codes, continuous)`:

            - `codes`: Discrete codes. Shape `(batch_size, num_codebooks, num_frames)`.
            - `continuous`: Continuous output:

                - If `domain` is `"x"`: Shape `(batch_size, emb_channels, num_frames)`.
                - If `domain` is `"q"`: Shape `(batch_size, emb_channels, num_frames)` if `no_sum` is `False` else
                    `(batch_size, emb_channels, num_codebooks, num_frames)`.
                - If `domain` is `"x_proj"`: Shape `(batch_size, codebook_channels, num_codebooks, num_frames)`.
                - If `domain` is `"q_proj"`: Shape `(batch_size, codebook_channels, num_codebooks, num_frames)`.
        """
        assert x.ndim == 3, f"Input must be 3-dimensional, got shape {x.shape}"
        assert x.shape[-1] % self.downsampling_factor == 0, (
            f"Input size along last dimension must be divisible by {self.downsampling_factor}. Got shape {x.shape}."
        )
        encoded = self.encoder(x)
        codes, quantized, _, _, x_proj, quantized_proj = self.quantizer(encoded, no_sum=no_sum)
        if domain == "x":
            return codes, encoded
        if domain == "q":
            return codes, quantized
        if domain == "x_proj":
            return codes, x_proj
        if domain == "q_proj":
            return codes, quantized_proj
        raise ValueError(f"Unknown domain: {domain}")

    def decode(self, x: torch.Tensor, no_sum: bool = False, domain: str = "code") -> torch.Tensor:
        """Decode input into audio.

        Args:
            x: Input tensor:

                - If `domain` is `"code"`: Shape `(batch_size, num_codebooks, num_frames)`.
                - If `domain` is `"x"`: Shape `(batch_size, emb_channels, num_frames)`.
                - If `domain` is `"q"`: Shape `(batch_size, emb_channels, num_frames)` if `no_sum` is `False` else
                    `(batch_size, emb_channels, num_codebooks, num_frames)`.
                - If `domain` is `"x_proj"`: Shape `(batch_size, codebook_channels, num_codebooks, num_frames)`.
                - If `domain` is `"q_proj"`: Shape `(batch_size, codebook_channels, num_codebooks, num_frames)`.
            no_sum: If `False`, the input quantized embeddings are assumed to be summed across codebooks. Ignored
                if `domain` is not `"q"`.
            domain: Domain of input tensor.

        Returns:
            Decoded audio. Shape `(batch_size, in_channels, num_samples)`.
        """
        quantized = self.quantizer.decode(x, input_no_sum=no_sum, domain=domain)
        return self.decoder(quantized)
