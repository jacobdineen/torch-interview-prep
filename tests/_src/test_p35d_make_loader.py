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
from p35d_make_loader import *

def test_p35d_make_loader():
    ds = SyntheticDataset(100, 4, seed=42)
    pass
    pass
    pass
    with step('make_loader produces correctly-shaped batches'):
        loader = make_loader(ds, batch_size=8, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (8, 4) and yb.shape == (8,)
    pass
