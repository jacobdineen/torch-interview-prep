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
from p09_einsum_practice import *

def test_p09_einsum_practice():

    torch.manual_seed(0)
    with step("matmul: (M,K) @ (K,N) -> (M,N)"):
        A = torch.randn(3, 4); B = torch.randn(4, 5)
        assert torch.allclose(matmul(A, B), A @ B, atol=1e-5)
    with step("batch_matmul broadcasts the batch dim"):
        A2 = torch.randn(2, 3, 4); B2 = torch.randn(2, 4, 5)
        assert torch.allclose(batch_matmul(A2, B2), A2 @ B2, atol=1e-5)
    with step("bilinear x^T W y computed per batch"):
        x = torch.randn(2, 3); W = torch.randn(3, 4); y = torch.randn(2, 4)
        expected = (x @ W * y).sum(dim=-1)
        assert torch.allclose(bilinear(x, W, y), expected, atol=1e-5)
    with step("attention_scores: q @ k^T over the head_dim"):
        q = torch.randn(2, 4, 6, 8); k = torch.randn(2, 4, 6, 8)
        scores = attention_scores(q, k)
        assert scores.shape == (2, 4, 6, 6)
        assert torch.allclose(scores, q @ k.transpose(-1, -2), atol=1e-4)

