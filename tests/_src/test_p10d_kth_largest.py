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
from p10d_kth_largest import *

def test_p10d_kth_largest():
    x = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0], [9.0, 0.0, 7.0, 8.0, 6.0]])
    pass
    pass
    pass
    with step('kth_largest picks the k-th largest (1-indexed) in a flat tensor'):
        flat = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        assert kth_largest(flat, 1) == 9
        assert kth_largest(flat, 3) == 5
        assert kth_largest(flat, 8) == 1
