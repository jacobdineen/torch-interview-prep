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
from p35a_dataset import *

def test_p35a_dataset():
    ds = SyntheticDataset(100, 4, seed=42)
    with step('dataset length matches num_samples'):
        assert len(ds) == 100
    pass
    pass
    pass
    pass
