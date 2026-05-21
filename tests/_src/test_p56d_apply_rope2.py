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
from p56d_apply_rope2 import *

def test_p56d_apply_rope2():
    torch.manual_seed(0)
    T, d = (16, 8)
    cos, sin = rope_freqs(T, d)
    pass
    pass
    B, H = (2, 4)
    q = torch.randn(B, H, T, d)
    qr = apply_rope(q, cos, sin)
    pass
    with step("apply_rope preserves L2 norm (it's a rotation)"):
        assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-05)
    pass
