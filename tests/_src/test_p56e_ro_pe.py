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
from p56e_ro_pe import *

def test_p56e_ro_pe():
    torch.manual_seed(0)
    T, d = (16, 8)
    cos, sin = rope_freqs(T, d)
    pass
    pass
    B, H = (2, 4)
    q = torch.randn(B, H, T, d)
    qr = apply_rope(q, cos, sin)
    pass
    pass
    with step('RoPE depends only on relative position (m - n)'):
        q1 = torch.randn(d)
        k1 = torch.randn(d)

        def dot(m, n):
            T_local = max(m, n) + 1
            cos_l, sin_l = rope_freqs(T_local, d)
            qm = apply_rope(q1.view(1, 1, 1, d), cos_l[m:m + 1], sin_l[m:m + 1]).view(d)
            kn = apply_rope(k1.view(1, 1, 1, d), cos_l[n:n + 1], sin_l[n:n + 1]).view(d)
            return (qm * kn).sum()
        d1 = dot(3, 5)
        d2 = dot(8, 10)
        assert torch.isclose(d1, d2, atol=1e-05)
