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
from p44b_my_conv2d2 import *

def test_p44b_my_conv2d2():
    torch.manual_seed(0)
    for stride, padding, kernel in [(1, 0, 3), (2, 1, 3), (1, 2, 5)]:
        pass
    with step('my_conv2d works with bias=None'):
        x = torch.randn(1, 1, 5, 5)
        w = torch.randn(2, 1, 3, 3)
        out = my_conv2d(x, w, bias=None, stride=1, padding=0)
        expected = F.conv2d(x, w, bias=None, stride=1, padding=0)
        assert torch.allclose(out, expected, atol=1e-05)
