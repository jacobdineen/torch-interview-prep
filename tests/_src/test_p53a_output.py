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
from p53a_output import *

def test_p53a_output():
    torch.manual_seed(0)
    B, T, D = (2, 5, 4)
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    with step('output shape (B, T, D) and attn shape (B, T, T)'):
        assert out.shape == (B, T, D)
        assert attn.shape == (B, T, T)
    pass
    pass
    pass
    pass
