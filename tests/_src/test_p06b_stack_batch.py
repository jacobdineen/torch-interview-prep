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
from p06b_stack_batch import *

def test_p06b_stack_batch():
    a = torch.arange(6).reshape(3, 2)
    b = torch.arange(9).reshape(3, 3) + 100
    pass
    with step('stack_batch creates a new leading dim'):
        items = [torch.ones(4) * i for i in range(5)]
        s = stack_batch(items)
        assert s.shape == (5, 4)
        assert torch.equal(s[2], torch.full((4,), 2.0))
    pass
    pass
