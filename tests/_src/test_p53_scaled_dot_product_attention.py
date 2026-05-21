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
from p53_scaled_dot_product_attention import *

def test_p53_scaled_dot_product_attention():

    torch.manual_seed(0)
    B, T, D = 2, 5, 4
    q = torch.randn(B, T, D); k = torch.randn(B, T, D); v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    with step("output shape (B, T, D) and attn shape (B, T, T)"):
        assert out.shape == (B, T, D)
        assert attn.shape == (B, T, T)
    with step("attention rows sum to 1"):
        assert torch.allclose(attn.sum(dim=-1), torch.ones(B, T), atol=1e-5)
    with step("matches torch.nn.functional.scaled_dot_product_attention"):
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(out, expected, atol=1e-5)
    with step("mask True positions get zero attention weight"):
        mask = torch.zeros(B, T, T, dtype=torch.bool)
        mask[:, 0, T // 2:] = True
        out_m, attn_m = scaled_dot_product_attention(q, k, v, mask=mask)
        assert torch.all(attn_m[:, 0, T // 2:] == 0)
        assert torch.allclose(attn_m.sum(dim=-1), torch.ones(B, T), atol=1e-5)
    with step("works with extra leading dim (B, H, T, D)"):
        q4 = torch.randn(B, 3, T, D); k4 = torch.randn(B, 3, T, D); v4 = torch.randn(B, 3, T, D)
        out4, _ = scaled_dot_product_attention(q4, k4, v4)
        assert out4.shape == (B, 3, T, D)

