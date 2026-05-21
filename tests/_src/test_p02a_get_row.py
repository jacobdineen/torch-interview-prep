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
from p02a_get_row import *

def test_p02a_get_row():
    x = torch.arange(20).reshape(4, 5)
    with step('get_row(x, 2) returns the 3rd row'):
        assert torch.equal(get_row(x, 2), torch.tensor([10, 11, 12, 13, 14]))
    sq = torch.arange(9).reshape(3, 3)
    pass
    pass
    pass
    pass
