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
from p54b_apply_causal_mask import *

def test_p54b_apply_causal_mask():
    pass
    with step('apply_causal_mask sets upper-triangle (strict) to -inf'):
        s = torch.zeros(2, 3, 4, 4)
        s2 = apply_causal_mask(s)
        assert torch.isinf(s2[0, 0, 0, 1]) and s2[0, 0, 0, 1] < 0
        assert s2[0, 0, 1, 1] == 0.0
        assert s2[0, 0, 2, 1] == 0.0
    torch.manual_seed(0)
    B, H, T, D = (2, 4, 6, 8)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out = causal_attention(q, k, v)
    pass
    pass
