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
from p63c_window import *

def test_p63c_window():
    T, W = (6, 3)
    mask = sliding_window_causal_mask(T, W)
    pass
    pass
    torch.manual_seed(0)
    B, H, D = (2, 4, 8)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    with step('window>=T equals causal attention'):
        out = sliding_window_attention(q, k, v, window_size=T + 5)
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        cm = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(cm, float('-inf'))
        ref = torch.softmax(scores, dim=-1) @ v
        assert torch.allclose(out, ref, atol=1e-05)
    pass
