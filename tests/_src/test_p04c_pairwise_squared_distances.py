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
from p04c_pairwise_squared_distances import *

def test_p04c_pairwise_squared_distances():
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    pass
    pass
    with step('pairwise_squared_distances computes ||a-b||^2 over D'):
        A = torch.randn(3, 4)
        B = torch.randn(2, 4)
        D = pairwise_squared_distances(A, B)
        assert D.shape == (3, 2)
        expected = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        assert torch.allclose(D, expected, atol=1e-05)
    pass
