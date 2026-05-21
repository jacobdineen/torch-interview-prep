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
import math
import torch
from p65c_stable import *

def test_p65c_stable():
    torch.manual_seed(0)

    def naive(q, k, v):
        D = q.shape[-1]
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        return torch.softmax(scores, dim=-1) @ v
    pass
    pass
    with step('stable at very large logits (online softmax)'):
        q = torch.randn(1, 1, 32, 8) * 10
        k = torch.randn(1, 1, 32, 8) * 10
        v = torch.randn(1, 1, 32, 8)
        out = flash_attention_tiled(q, k, v, block_size_q=8, block_size_k=8)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, naive(q, k, v), atol=0.001)
