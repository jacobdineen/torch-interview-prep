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
from p55a_output import *

def test_p55a_output():
    torch.manual_seed(0)
    B, T, D, H = (2, 7, 16, 4)
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    out = mha(x)
    with step('output shape matches input'):
        assert out.shape == (B, T, D)
    pass
    pass
    pass
