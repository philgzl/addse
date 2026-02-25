from contextlib import AbstractContextManager, nullcontext

import pytest

from addse.utils import bytes_str_to_int


@pytest.mark.parametrize(
    ("bytes_str", "expected", "context"),
    [
        ("1KB", 1000, nullcontext()),
        ("16MB", 16 * 1000**2, nullcontext()),
        ("256GB", 256 * 1000**3, nullcontext()),
        ("4TB", 4 * 1000**4, nullcontext()),
        ("8B", None, pytest.raises(ValueError, match="Invalid size")),
        ("32", None, pytest.raises(ValueError, match="Invalid size")),
        ("KB", None, pytest.raises(ValueError, match="Invalid size")),
        ("1PB", None, pytest.raises(ValueError, match="Invalid size")),
    ],
)
def test_bytes_str_to_int(bytes_str: str, expected: int, context: AbstractContextManager) -> None:
    """Test human-readable byte size conversion."""
    with context:
        output = bytes_str_to_int(bytes_str)
        assert output == expected, (output, expected)
