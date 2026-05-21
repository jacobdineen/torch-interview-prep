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
from p09c_bilinear import *

def test_p09c_bilinear():
    torch.manual_seed(0)
    pass
    pass
    with step('bilinear x^T W y computed per batch'):
        x = torch.randn(2, 3)
        W = torch.randn(3, 4)
        y = torch.randn(2, 4)
        expected = (x @ W * y).sum(dim=-1)
        assert torch.allclose(bilinear(x, W, y), expected, atol=1e-05)
    pass
