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
from p02b_get_diagonal import *

def test_p02b_get_diagonal():
    x = torch.arange(20).reshape(4, 5)
    pass
    sq = torch.arange(9).reshape(3, 3)
    with step('get_diagonal returns the matrix diagonal'):
        assert torch.equal(get_diagonal(sq), torch.tensor([0, 4, 8]))
    pass
    pass
    pass
