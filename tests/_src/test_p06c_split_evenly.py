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
from p06c_split_evenly import *

def test_p06c_split_evenly():
    a = torch.arange(6).reshape(3, 2)
    b = torch.arange(9).reshape(3, 3) + 100
    pass
    pass
    with step('split_evenly partitions into k equal chunks'):
        x = torch.arange(12)
        parts = split_evenly(x, 4)
        assert len(parts) == 4
        assert all((p.shape == (3,) for p in parts))
        assert torch.equal(torch.cat(list(parts)), x)
    pass
