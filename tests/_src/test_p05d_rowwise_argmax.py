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
from p05d_rowwise_argmax import *

def test_p05d_rowwise_argmax():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 16, 16)
    pass
    pass
    pass
    with step('rowwise_argmax returns the index of max in each row'):
        a = torch.tensor([[1.0, 3.0, 2.0], [9.0, 0.0, 5.0]])
        assert torch.equal(rowwise_argmax(a), torch.tensor([1, 0]))
