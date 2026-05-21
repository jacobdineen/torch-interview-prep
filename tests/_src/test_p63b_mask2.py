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
import math
import torch
from p63b_mask2 import *

def test_p63b_mask2():
    T, W = (6, 3)
    mask = sliding_window_causal_mask(T, W)
    pass
    with step('mask pattern: unmasked iff j <= i AND i - j < W'):
        expected = torch.ones(T, T, dtype=torch.bool)
        for i in range(T):
            for j in range(max(0, i - W + 1), i + 1):
                expected[i, j] = False
        assert torch.equal(mask, expected)
    torch.manual_seed(0)
    B, H, D = (2, 4, 8)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    pass
    pass
