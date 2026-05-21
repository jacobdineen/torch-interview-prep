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
from p09d_attention_scores import *

def test_p09d_attention_scores():
    torch.manual_seed(0)
    pass
    pass
    pass
    with step('attention_scores: q @ k^T over the head_dim'):
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 4, 6, 8)
        scores = attention_scores(q, k)
        assert scores.shape == (2, 4, 6, 6)
        assert torch.allclose(scores, q @ k.transpose(-1, -2), atol=0.0001)
