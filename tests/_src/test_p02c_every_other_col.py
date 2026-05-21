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
from p02c_every_other_col import *

def test_p02c_every_other_col():
    x = torch.arange(20).reshape(4, 5)
    pass
    sq = torch.arange(9).reshape(3, 3)
    pass
    with step('every_other_col returns columns 0, 2, 4, ...'):
        assert torch.equal(every_other_col(x), x[:, ::2])
    pass
    pass
