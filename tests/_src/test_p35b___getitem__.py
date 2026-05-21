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
from p35b___getitem__ import *

def test_p35b___getitem__():
    ds = SyntheticDataset(100, 4, seed=42)
    pass
    with step('__getitem__ returns (x, y) of right shape and y = sum(x^2)'):
        x, y = ds[0]
        assert x.shape == (4,) and y.shape == ()
        assert torch.isclose(y, (x ** 2).sum())
    pass
    pass
    pass
