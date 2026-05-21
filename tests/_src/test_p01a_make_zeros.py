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
from p01a_make_zeros import *

def test_p01a_make_zeros():
    with step('make_zeros returns the requested shape and dtype'):
        z = make_zeros((2, 3), dtype=torch.float64)
        assert tuple(z.shape) == (2, 3)
        assert z.dtype == torch.float64
        assert torch.equal(z, torch.zeros(2, 3, dtype=torch.float64))
    pass
    pass
    pass
