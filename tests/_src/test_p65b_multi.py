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
from p65b_multi import *

def test_p65b_multi():
    torch.manual_seed(0)

    def naive(q, k, v):
        D = q.shape[-1]
        scores = q @ k.transpose(-1, -2) / math.sqrt(D)
        return torch.softmax(scores, dim=-1) @ v
    pass
    with step('multi-block with T not divisible by block sizes'):
        q = torch.randn(2, 4, 67, 32)
        k = torch.randn(2, 4, 67, 32)
        v = torch.randn(2, 4, 67, 32)
        out = flash_attention_tiled(q, k, v, block_size_q=16, block_size_k=24)
        assert torch.allclose(out, naive(q, k, v), atol=0.0001)
    pass
