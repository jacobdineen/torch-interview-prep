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
from p54_causal_mask import *

def test_p54_causal_mask():

    with step("causal_mask(5) is True above diagonal, False on/below"):
        m = causal_mask(5)
        assert m.shape == (5, 5) and m.dtype == torch.bool
        expected = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
        assert torch.equal(m, expected)
    with step("apply_causal_mask sets upper-triangle (strict) to -inf"):
        s = torch.zeros(2, 3, 4, 4)
        s2 = apply_causal_mask(s)
        assert torch.isinf(s2[0, 0, 0, 1]) and s2[0, 0, 0, 1] < 0
        assert s2[0, 0, 1, 1] == 0.0
        assert s2[0, 0, 2, 1] == 0.0
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 6, 8
    q = torch.randn(B, H, T, D); k = torch.randn(B, H, T, D); v = torch.randn(B, H, T, D)
    out = causal_attention(q, k, v)
    with step("causal_attention output shape (B, H, T, D)"):
        assert out.shape == (B, H, T, D)
    with step("perturbing future tokens (k, v at positions >= 4) leaves earlier outputs unchanged"):
        k2 = k.clone(); k2[:, :, 4:] = torch.randn_like(k2[:, :, 4:])
        v2 = v.clone(); v2[:, :, 4:] = torch.randn_like(v2[:, :, 4:])
        out2 = causal_attention(q, k2, v2)
        assert torch.allclose(out[:, :, :4], out2[:, :, :4], atol=1e-5)
        assert not torch.allclose(out[:, :, 4:], out2[:, :, 4:], atol=1e-5)

