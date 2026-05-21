import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import torch
from p56_rope_embeddings import *

def test_p56_rope_embeddings():

    torch.manual_seed(0)
    T, d = 16, 8
    cos, sin = rope_freqs(T, d)
    with step("rope_freqs returns (T, d/2) cos and sin"):
        assert cos.shape == (T, d // 2)
        assert sin.shape == (T, d // 2)
    with step("position 0 has cos=1, sin=0"):
        assert torch.allclose(cos[0], torch.ones(d // 2), atol=1e-6)
        assert torch.allclose(sin[0], torch.zeros(d // 2), atol=1e-6)
    B, H = 2, 4
    q = torch.randn(B, H, T, d)
    qr = apply_rope(q, cos, sin)
    with step("apply_rope is shape-preserving"):
        assert qr.shape == q.shape
    with step("apply_rope preserves L2 norm (it's a rotation)"):
        assert torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-5)
    with step("RoPE depends only on relative position (m - n)"):
        q1 = torch.randn(d); k1 = torch.randn(d)
        def dot(m, n):
            T_local = max(m, n) + 1
            cos_l, sin_l = rope_freqs(T_local, d)
            qm = apply_rope(q1.view(1, 1, 1, d), cos_l[m:m + 1], sin_l[m:m + 1]).view(d)
            kn = apply_rope(k1.view(1, 1, 1, d), cos_l[n:n + 1], sin_l[n:n + 1]).view(d)
            return (qm * kn).sum()
        d1 = dot(3, 5); d2 = dot(8, 10)
        assert torch.isclose(d1, d2, atol=1e-5)

