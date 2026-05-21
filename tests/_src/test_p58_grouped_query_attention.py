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
import torch.nn as nn
from p58_grouped_query_attention import *

def test_p58_grouped_query_attention():

    torch.manual_seed(0)
    D, H, G = 32, 8, 4
    gqa = GroupedQueryAttention(D, num_q_heads=H, num_kv_groups=G, bias=False)
    x = torch.randn(2, 5, D)
    out = gqa(x)
    head_dim = D // H
    with step("output shape matches input"):
        assert out.shape == (2, 5, D)
    with step("w_q weight is (H*head_dim, D); w_k/w_v are (G*head_dim, D)"):
        assert gqa.w_q.weight.shape == (H * head_dim, D)
        assert gqa.w_k.weight.shape == (G * head_dim, D)
        assert gqa.w_v.weight.shape == (G * head_dim, D)
        assert gqa.w_o.weight.shape == (D, H * head_dim)
    with step("MQA degenerate (G=1) still works"):
        mqa = GroupedQueryAttention(D, num_q_heads=H, num_kv_groups=1, bias=False)
        assert mqa(x).shape == (2, 5, D)
    with step("MHA degenerate (G=H) still works"):
        mha_like = GroupedQueryAttention(D, num_q_heads=H, num_kv_groups=H, bias=False)
        assert mha_like(x).shape == (2, 5, D)
    with step("accepts a (T, T) causal mask"):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out_m = gqa(x, mask=mask)
        assert out_m.shape == (2, T, D)
    with step("gradients flow into every parameter"):
        loss = out.sum(); loss.backward()
        for name, p in gqa.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

