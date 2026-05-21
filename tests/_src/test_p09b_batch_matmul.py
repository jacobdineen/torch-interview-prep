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
from p09b_batch_matmul import *

def test_p09b_batch_matmul():
    torch.manual_seed(0)
    pass
    with step('batch_matmul broadcasts the batch dim'):
        A2 = torch.randn(2, 3, 4)
        B2 = torch.randn(2, 4, 5)
        assert torch.allclose(batch_matmul(A2, B2), A2 @ B2, atol=1e-05)
    pass
    pass
