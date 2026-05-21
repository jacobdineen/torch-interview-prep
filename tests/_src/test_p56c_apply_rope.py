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
from p56c_apply_rope import *

def test_p56c_apply_rope():
    torch.manual_seed(0)
    T, d = (16, 8)
    cos, sin = rope_freqs(T, d)
    pass
    pass
    B, H = (2, 4)
    q = torch.randn(B, H, T, d)
    qr = apply_rope(q, cos, sin)
    with step('apply_rope is shape-preserving'):
        assert qr.shape == q.shape
    pass
    pass
