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
from p27d_p import *

def test_p27d_p():
    torch.manual_seed(0)
    p = 0.4
    d = MyDropout(p)
    x = torch.ones(10000)
    pass
    pass
    pass
    with step('p=0 is identity in train mode'):
        d0 = MyDropout(0.0)
        d0.train()
        assert torch.equal(d0(x), x)
