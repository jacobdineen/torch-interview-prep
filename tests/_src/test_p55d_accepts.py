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
import torch.nn as nn
from p55d_accepts import *

def test_p55d_accepts():
    torch.manual_seed(0)
    B, T, D, H = (2, 7, 16, 4)
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    out = mha(x)
    pass
    pass
    pass
    with step('accepts a (T, T) causal mask'):
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out_causal = mha(x, mask=mask)
        assert out_causal.shape == (B, T, D)
