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
from p07b_count_above import *

def test_p07b_count_above():
    x = torch.tensor([-1.0, 0.0, 2.0, -3.0, 4.0])
    pass
    with step('count_above counts strictly-greater entries'):
        assert count_above(x, 0.0) == 2
    pass
    pass
