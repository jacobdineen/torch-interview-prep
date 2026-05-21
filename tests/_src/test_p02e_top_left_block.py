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
from p02e_top_left_block import *

def test_p02e_top_left_block():
    x = torch.arange(20).reshape(4, 5)
    pass
    sq = torch.arange(9).reshape(3, 3)
    pass
    pass
    pass
    with step('top_left_block(x, 2) returns the top-left 2x2'):
        assert torch.equal(top_left_block(x, 2), x[:2, :2])
