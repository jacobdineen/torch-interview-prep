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
from p54a_causal_mask import *

def test_p54a_causal_mask():
    with step('causal_mask(5) is True above diagonal, False on/below'):
        m = causal_mask(5)
        assert m.shape == (5, 5) and m.dtype == torch.bool
        expected = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        assert torch.equal(m, expected)
    pass
    torch.manual_seed(0)
    B, H, T, D = (2, 4, 6, 8)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out = causal_attention(q, k, v)
    pass
    pass
