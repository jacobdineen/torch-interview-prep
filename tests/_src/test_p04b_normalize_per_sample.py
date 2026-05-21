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
from p04b_normalize_per_sample import *

def test_p04b_normalize_per_sample():
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    pass
    with step('normalize_per_sample yields per-row mean 0 std 1'):
        n = normalize_per_sample(X)
        assert torch.allclose(n.mean(dim=1), torch.zeros(5), atol=1e-05)
        assert torch.allclose(n.std(dim=1, unbiased=False), torch.ones(5), atol=0.0001)
    pass
    pass
