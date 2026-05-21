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
from p08b_one_hot import *

def test_p08b_one_hot():
    pass
    with step('one_hot makes a (N, C) float matrix'):
        labels = torch.tensor([0, 2, 1, 2])
        oh = one_hot(labels, num_classes=3)
        assert oh.shape == (4, 3)
        assert oh.dtype.is_floating_point
        assert torch.equal(oh, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    pass
