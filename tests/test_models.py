import pytest
import torch
import torch.nn as nn

from addse.models import (
    BSRNN,
    NAC,
    ADDSERQDiT,
    ConvTasNet,
    MPDiscriminator,
    MSSTFTDiscriminator,
    SGMSEUNet,
)


@pytest.mark.parametrize("model_cls", [BSRNN, ConvTasNet])
@pytest.mark.parametrize("input_channels", [1, 2])
@pytest.mark.parametrize("output_channels", [1, 2])
@torch.no_grad()
def test_model_forward(
    torch_rng: torch.Generator, model_cls: type[nn.Module], input_channels: int, output_channels: int
) -> None:
    """Test model forward pass."""
    batch_size, samples = 4, 1000
    batch = torch.randn(batch_size, input_channels, samples, generator=torch_rng)
    model = model_cls(input_channels=input_channels, output_channels=output_channels).eval()
    output = model(batch)
    expected_shape = (batch_size, output_channels, samples)
    assert output.shape == expected_shape, (output.shape, expected_shape)
    assert output.dtype == batch.dtype, (output.dtype, batch.dtype)


@pytest.mark.parametrize(("model_cls", "latency"), [(BSRNN, 512), (ConvTasNet, 32)])
@pytest.mark.parametrize("shape", [(1, 1, 2048)])
@torch.no_grad()
def test_model_latency(
    torch_rng: torch.Generator, model_cls: type[nn.Module], latency: int, shape: tuple[int, ...]
) -> None:
    """Test model latency."""
    model = model_cls(causal=True).eval()
    # test latency check passes for latency=latency
    latency_check(torch_rng, model, latency, shape)
    # test latency check fails for latency=latency - 1
    with pytest.raises(AssertionError):
        latency_check(torch_rng, model, latency - 1, shape)
    # test latency check fails when initializing model with causal=False
    model = model_cls(causal=False).eval()
    with pytest.raises(AssertionError):
        latency_check(torch_rng, model, latency, shape)


def latency_check(torch_rng: torch.Generator, model: nn.Module, latency: int, shape: tuple[int, ...]) -> None:
    """Perform latency check."""
    # for start in range(shape[-1] - latency):
    for start in [256]:
        batch = torch.randn(shape, generator=torch_rng)
        batch[..., start + latency :] = float("nan")
        output = model(batch)
        assert next(k for k in range(shape[-1]) if output[..., k].isnan().any()) > start


@torch.no_grad()
def test_msstftd_discriminator_forward(torch_rng: torch.Generator) -> None:
    """Test MSSTFTDiscriminator forward pass."""
    batch_size, in_channels, samples = 4, 1, 16384
    frame_lengths = [256, 512, 1024]
    batch = torch.randn(batch_size, in_channels, samples, generator=torch_rng)
    model = MSSTFTDiscriminator(frame_lengths=frame_lengths, in_channels=in_channels, num_channels=1).eval()
    outputs, featuress = model(batch)
    assert isinstance(outputs, list)
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert all(output.dtype == batch.dtype for output in outputs)
    assert len(outputs) == len(frame_lengths)
    assert all(output.ndim == 4 for output in outputs)
    assert all(output.shape[0] == batch_size for output in outputs)
    assert all(output.shape[1] == 1 for output in outputs)
    assert isinstance(featuress, list)
    assert all(isinstance(features, list) for features in featuress)
    assert all(all(isinstance(feature, torch.Tensor) for feature in features) for features in featuress)
    assert all(all(feature.dtype == batch.dtype for feature in features) for features in featuress)
    assert len(featuress) == len(frame_lengths)
    assert all(all(feature.ndim == 4 for feature in features) for features in featuress)
    assert all(all(feature.shape[0] == batch_size for feature in features) for features in featuress)


@torch.no_grad()
def test_mp_discriminator_forward(torch_rng: torch.Generator) -> None:
    """Test MPDiscriminator forward pass."""
    batch_size, in_channels, samples = 4, 1, 16384
    periods = [2, 3, 5, 7, 11]
    batch = torch.randn(batch_size, in_channels, samples, generator=torch_rng)
    model = MPDiscriminator(periods=periods, in_channels=in_channels, channels=[1, 1, 1, 1, 1]).eval()
    outputs, featuress = model(batch)
    assert isinstance(outputs, list)
    assert all(isinstance(output, torch.Tensor) for output in outputs)
    assert all(output.dtype == batch.dtype for output in outputs)
    assert len(outputs) == len(periods)
    assert all(output.ndim == 4 for output in outputs)
    assert all(output.shape[0] == batch_size for output in outputs)
    assert all(output.shape[1] == 1 for output in outputs)
    assert isinstance(featuress, list)
    assert all(isinstance(features, list) for features in featuress)
    assert all(all(isinstance(feature, torch.Tensor) for feature in features) for features in featuress)
    assert all(all(feature.dtype == batch.dtype for feature in features) for features in featuress)
    assert len(featuress) == len(periods)
    assert all(all(feature.ndim == 4 for feature in features) for features in featuress)
    assert all(all(feature.shape[0] == batch_size for feature in features) for features in featuress)


@pytest.mark.parametrize("normalize", [True, False])
@torch.no_grad()
def test_nac_forward(torch_rng: torch.Generator, normalize: bool) -> None:
    """Test NAC forward pass."""
    batch_size, in_channels, samples = 2, 1, 320
    num_codebooks = 4
    batch = torch.randn(batch_size, in_channels, samples, generator=torch_rng)
    model = NAC(in_channels, 3, 5, codebook_size=7, num_codebooks=num_codebooks, normalize=normalize).eval()
    reconstructed, codes, codebook_loss, commit_loss = model(batch)
    assert reconstructed.shape == batch.shape
    assert reconstructed.dtype == batch.dtype
    assert codes.ndim == 3
    assert codes.shape == (batch_size, num_codebooks, samples // model.downsampling_factor)
    assert codes.dtype == torch.long
    assert codebook_loss.ndim == 0
    assert codebook_loss.dtype == batch.dtype
    assert commit_loss.ndim == 0
    assert commit_loss.dtype == batch.dtype


@pytest.mark.parametrize("domain", ["x", "q", "x_proj", "q_proj"])
@pytest.mark.parametrize("no_sum", [False, True])
@torch.no_grad()
def test_nac_encode(torch_rng: torch.Generator, no_sum: bool, domain: str) -> None:
    """Test NAC encode method."""
    batch_size, in_channels, samples = 2, 1, 320
    num_codebooks = 4
    emb_channels = 3
    base_channes = 5
    codebook_channels = 7
    codebook_size = 11
    batch = torch.randn(batch_size, in_channels, samples, generator=torch_rng)
    model = NAC(
        in_channels,
        emb_channels,
        base_channes,
        codebook_size=codebook_size,
        codebook_channels=codebook_channels,
        num_codebooks=num_codebooks,
    ).eval()
    codes, continuous = model.encode(batch, no_sum=no_sum, domain=domain)
    assert codes.ndim == 3
    assert codes.shape == (batch_size, num_codebooks, samples // model.downsampling_factor)
    assert codes.dtype == torch.long
    if domain == "x":
        assert continuous.ndim == 3
        assert continuous.shape == (batch_size, emb_channels, samples // model.downsampling_factor)
    elif domain == "q":
        assert continuous.ndim == (4 if no_sum else 3)
        assert continuous.shape == (
            (batch_size, emb_channels, num_codebooks, samples // model.downsampling_factor)
            if no_sum
            else (batch_size, emb_channels, samples // model.downsampling_factor)
        )
    elif domain in ("x_proj", "q_proj"):
        assert continuous.ndim == 4
        assert continuous.shape == (batch_size, codebook_channels, num_codebooks, samples // model.downsampling_factor)
    assert continuous.dtype == batch.dtype


@pytest.mark.parametrize("domain", ["x", "q", "x_proj", "q_proj"])
@pytest.mark.parametrize("no_sum", [False, True])
@torch.no_grad()
def test_nac_decode(torch_rng: torch.Generator, no_sum: bool, domain: str) -> None:
    """Test NAC decode method."""
    batch_size, in_channels, samples = 2, 1, 320
    num_codebooks = 4
    emb_channels = 3
    base_channes = 5
    codebook_channels = 7
    codebook_size = 11
    model = NAC(
        in_channels,
        emb_channels,
        base_channes,
        codebook_size=codebook_size,
        codebook_channels=codebook_channels,
        num_codebooks=num_codebooks,
    ).eval()
    if domain == "x":
        batch = torch.randn(batch_size, emb_channels, samples // model.downsampling_factor, generator=torch_rng)
    elif domain == "q":
        if no_sum:
            batch = torch.randn(
                batch_size,
                emb_channels,
                num_codebooks,
                samples // model.downsampling_factor,
                generator=torch_rng,
            )
        else:
            batch = torch.randn(batch_size, emb_channels, samples // model.downsampling_factor, generator=torch_rng)
    elif domain in ("x_proj", "q_proj"):
        batch = torch.randn(
            batch_size,
            codebook_channels,
            num_codebooks,
            samples // model.downsampling_factor,
            generator=torch_rng,
        )
    audio = model.decode(batch, no_sum=no_sum, domain=domain)
    assert audio.ndim == 3
    assert audio.shape == (batch_size, in_channels, samples)
    assert audio.dtype == torch.float


@pytest.mark.parametrize("domain", ["x", "q", "x_proj", "q_proj"])
@pytest.mark.parametrize("no_sum", [False, True])
@torch.no_grad()
def test_nac_encode_decode(torch_rng: torch.Generator, no_sum: bool, domain: str) -> None:
    """Test NAC encode-decode pipeline."""
    batch_size, in_channels, samples = 2, 1, 320
    num_codebooks = 4
    emb_channels = 3
    base_channes = 5
    codebook_channels = 7
    codebook_size = 11
    batch = torch.randn(batch_size, in_channels, samples, generator=torch_rng)
    model = NAC(
        in_channels,
        emb_channels,
        base_channes,
        codebook_size=codebook_size,
        codebook_channels=codebook_channels,
        num_codebooks=num_codebooks,
    ).eval()
    codes, continuous = model.encode(batch, no_sum=no_sum, domain=domain)
    audio = (
        model.decode(codes, no_sum=no_sum, domain=domain)
        if domain == "code"
        else model.decode(continuous, no_sum=no_sum, domain=domain)
    )
    assert audio.ndim == 3
    assert audio.shape == (batch_size, in_channels, samples)
    assert audio.dtype == batch.dtype


@torch.no_grad()
def test_addse_rqdit_forward(torch_rng: torch.Generator) -> None:
    """Test ADDSERQDiT forward pass."""
    batch_size, input_channels, num_codebooks, samples = 1, 3, 5, 7
    output_channels = 1
    model = ADDSERQDiT(
        input_channels=input_channels,
        output_channels=output_channels,
        num_codebooks=num_codebooks,
        hidden_dim=6,
        num_layers=4,
        num_heads=3,
        max_seq_len=samples,
        conditional=True,
        time_independent=True,
    ).eval()
    x = torch.randn(batch_size, input_channels, num_codebooks, samples, generator=torch_rng)
    c = torch.randn(batch_size, input_channels, num_codebooks, samples, generator=torch_rng)
    output = model(x, c)
    expected_shape = (batch_size, output_channels, num_codebooks, samples)
    assert output.shape == expected_shape, (output.shape, expected_shape)
    assert output.dtype == x.dtype


@torch.no_grad()
def test_sgmse_unet_forward(torch_rng: torch.Generator) -> None:
    """Test SGMSEUNet forward pass."""
    batch_size, num_channels, freqs, frames = 1, 1, 8, 8
    model = SGMSEUNet(num_channels, base_channels=4, channel_mult=(1, 1, 1, 1)).eval()
    x = torch.randn(batch_size, num_channels, freqs, frames, generator=torch_rng, dtype=torch.cfloat)
    y = torch.randn(batch_size, num_channels, freqs, frames, generator=torch_rng, dtype=torch.cfloat)
    emb = torch.randn(batch_size, generator=torch_rng)
    output = model(x, y, emb)
    expected_shape = (batch_size, num_channels, freqs, frames)
    assert output.shape == expected_shape, (output.shape, expected_shape)
    assert output.dtype == x.dtype, (output.dtype, x.dtype)
