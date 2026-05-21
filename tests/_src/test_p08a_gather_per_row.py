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
from p08a_gather_per_row import *

def test_p08a_gather_per_row():
    with step('gather_per_row picks one column per row'):
        x = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]])
        idx = torch.tensor([2, 0, 1])
        assert torch.equal(gather_per_row(x, idx), torch.tensor([30.0, 40.0, 80.0]))
    pass
    pass
