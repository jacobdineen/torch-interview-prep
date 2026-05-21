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
from p06d_interleave import *

def test_p06d_interleave():
    a = torch.arange(6).reshape(3, 2)
    b = torch.arange(9).reshape(3, 3) + 100
    pass
    pass
    pass
    with step('interleave alternates rows from a and b'):
        a2 = torch.tensor([[1, 1], [3, 3]])
        b2 = torch.tensor([[2, 2], [4, 4]])
        result = interleave(a2, b2)
        assert torch.equal(result, torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]]))
