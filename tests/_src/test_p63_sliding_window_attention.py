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

import math
import torch
from p63_sliding_window_attention import *

def test_p63_sliding_window_attention():

    T, W = 6, 3
    mask = sliding_window_causal_mask(T, W)
    with step("mask shape and dtype"):
        assert mask.shape == (T, T) and mask.dtype == torch.bool
    with step("mask pattern: unmasked iff j <= i AND i - j < W"):
        expected = torch.ones(T, T, dtype=torch.bool)
        for i in range(T):
            for j in range(max(0, i - W + 1), i + 1):
                expected[i, j] = False
        assert torch.equal(mask, expected)
    torch.manual_seed(0)
    B, H, D = 2, 4, 8
    q = torch.randn(B, H, T, D); k = torch.randn(B, H, T, D); v = torch.randn(B, H, T, D)
    with step("window>=T equals causal attention"):
        out = sliding_window_attention(q, k, v, window_size=T + 5)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
        cm = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(cm, float("-inf"))
        ref = torch.softmax(scores, dim=-1) @ v
        assert torch.allclose(out, ref, atol=1e-5)
    with step("window=1: output equals v (each token sees only itself)"):
        out_w1 = sliding_window_attention(q, k, v, window_size=1)
        assert torch.allclose(out_w1, v, atol=1e-5)

