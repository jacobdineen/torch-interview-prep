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
from p35c_seeded import *

def test_p35c_seeded():
    ds = SyntheticDataset(100, 4, seed=42)
    pass
    pass
    with step('seeded dataset is deterministic'):
        ds2 = SyntheticDataset(100, 4, seed=42)
        x2, y2 = ds2[0]
        x, y = ds[0]
        assert torch.equal(x, x2) and torch.equal(y, y2)
    pass
    pass
