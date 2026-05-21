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
from p63d_window2 import *

def test_p63d_window2():
    T, W = (6, 3)
    mask = sliding_window_causal_mask(T, W)
    pass
    pass
    torch.manual_seed(0)
    B, H, D = (2, 4, 8)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    pass
    with step('window=1: output equals v (each token sees only itself)'):
        out_w1 = sliding_window_attention(q, k, v, window_size=1)
        assert torch.allclose(out_w1, v, atol=1e-05)
