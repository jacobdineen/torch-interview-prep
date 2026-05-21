import contextlib


@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError with the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import contextlib
import torch
import torch.nn.functional as F
from p44a_my_conv2d import *

def test_p44a_my_conv2d():
    torch.manual_seed(0)
    for stride, padding, kernel in [(1, 0, 3), (2, 1, 3), (1, 2, 5)]:
        with step(f'my_conv2d matches F.conv2d (stride={stride}, padding={padding}, kernel={kernel})'):
            x = torch.randn(2, 3, 16, 16)
            w = torch.randn(8, 3, kernel, kernel)
            b = torch.randn(8)
            out = my_conv2d(x, w, b, stride=stride, padding=padding)
            expected = F.conv2d(x, w, b, stride=stride, padding=padding)
            assert out.shape == expected.shape
            assert torch.allclose(out, expected, atol=0.0001)
    pass
