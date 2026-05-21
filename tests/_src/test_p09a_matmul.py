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
from p09a_matmul import *

def test_p09a_matmul():
    torch.manual_seed(0)
    with step('matmul: (M,K) @ (K,N) -> (M,N)'):
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        assert torch.allclose(matmul(A, B), A @ B, atol=1e-05)
    pass
    pass
    pass
