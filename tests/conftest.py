import io

import numpy as np
import pytest
import soundfile as sf
import torch
from litdata.streaming.cache import Cache

fs = 8000


@pytest.fixture
def torch_rng() -> torch.Generator:
    """Seeded PyTorch random number generator."""
    return torch.Generator().manual_seed(42)


@pytest.fixture(scope="session")
def input_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a dummy directory with data."""
    tmpdir = tmp_path_factory.mktemp("data")
    cache = Cache(input_dir=str(tmpdir), chunk_size=2)
    rng = np.random.default_rng(42)
    for i in range(10):
        n = fs if i % 2 == 0 else int(1.2 * fs)
        x = rng.standard_normal(n)
        buffer = io.BytesIO()
        sf.write(buffer, x, fs, format="wav")
        buffer.seek(0)
        cache[i] = buffer.read(), i
    cache.done()
    cache.merge()
    return str(tmpdir)


@pytest.fixture(scope="session")
def input_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Create two dummy directories with data."""
    tmpdir1 = tmp_path_factory.mktemp("data1")
    tmpdir2 = tmp_path_factory.mktemp("data2")
    rng = np.random.default_rng(42)
    for tmpdir in [tmpdir1, tmpdir2]:
        cache = Cache(input_dir=str(tmpdir), chunk_size=2)
        for i in range(10):
            n = fs if i % 2 == 0 else int(1.2 * fs)
            x = rng.standard_normal(n)
            buffer = io.BytesIO()
            sf.write(buffer, x, fs, format="wav")
            buffer.seek(0)
            cache[i] = buffer.read(), i
        cache.done()
        cache.merge()
    return str(tmpdir1), str(tmpdir2)
