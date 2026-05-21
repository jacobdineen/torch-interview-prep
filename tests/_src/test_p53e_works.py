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
from p53e_works import *

def test_p53e_works():
    torch.manual_seed(0)
    B, T, D = (2, 5, 4)
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    pass
    pass
    pass
    pass
    with step('works with extra leading dim (B, H, T, D)'):
        q4 = torch.randn(B, 3, T, D)
        k4 = torch.randn(B, 3, T, D)
        v4 = torch.randn(B, 3, T, D)
        out4, _ = scaled_dot_product_attention(q4, k4, v4)
        assert out4.shape == (B, 3, T, D)
