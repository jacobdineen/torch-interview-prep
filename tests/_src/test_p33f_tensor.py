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
import math
import torch
from p33f_tensor import *

def test_p33f_tensor():
    pass
    pass
    pass
    pass
    pass
    with step('tensor input returns tensor of same shape'):
        steps = torch.tensor([0, 5, 10, 55, 100, 500])
        lrs = linear_warmup_cosine_lr(steps, 1.0, 10, 100, min_lr=0.1)
        assert lrs.shape == steps.shape
        expected = torch.tensor([0.0, 0.5, 1.0, 0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * 0.5)), 0.1, 0.1])
        assert torch.allclose(lrs.float(), expected.float(), atol=1e-06)
