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
from p27c_eval import *

def test_p27c_eval():
    torch.manual_seed(0)
    p = 0.4
    d = MyDropout(p)
    x = torch.ones(10000)
    pass
    pass
    with step('eval mode is identity'):
        d.eval()
        y2 = d(x)
        assert torch.equal(y2, x)
    pass
