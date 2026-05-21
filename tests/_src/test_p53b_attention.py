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
from p53b_attention import *

def test_p53b_attention():
    torch.manual_seed(0)
    B, T, D = (2, 5, 4)
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    pass
    with step('attention rows sum to 1'):
        assert torch.allclose(attn.sum(dim=-1), torch.ones(B, T), atol=1e-05)
    pass
    pass
    pass
