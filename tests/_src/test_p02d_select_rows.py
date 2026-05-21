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
from p02d_select_rows import *

def test_p02d_select_rows():
    x = torch.arange(20).reshape(4, 5)
    pass
    sq = torch.arange(9).reshape(3, 3)
    pass
    pass
    with step('select_rows picks rows by index'):
        idx = torch.tensor([3, 0])
        assert torch.equal(select_rows(x, idx), x[[3, 0]])
    pass
