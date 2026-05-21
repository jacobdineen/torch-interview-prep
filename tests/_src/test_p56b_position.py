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
from p56b_position import *

def test_p56b_position():
    torch.manual_seed(0)
    T, d = (16, 8)
    cos, sin = rope_freqs(T, d)
    pass
    with step('position 0 has cos=1, sin=0'):
        assert torch.allclose(cos[0], torch.ones(d // 2), atol=1e-06)
        assert torch.allclose(sin[0], torch.zeros(d // 2), atol=1e-06)
    B, H = (2, 4)
    q = torch.randn(B, H, T, d)
    qr = apply_rope(q, cos, sin)
    pass
    pass
    pass
