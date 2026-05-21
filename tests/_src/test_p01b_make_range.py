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
from p01b_make_range import *

def test_p01b_make_range():
    pass
    with step('make_range returns a half-open arithmetic range'):
        r = make_range(0, 10, 2)
        assert torch.equal(r, torch.tensor([0, 2, 4, 6, 8]))
    pass
    pass
