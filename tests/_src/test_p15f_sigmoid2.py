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
from p15f_sigmoid2 import *

def test_p15f_sigmoid2():
    torch.manual_seed(0)
    x = torch.randn(50) * 5
    pass
    pass
    pass
    pass
    pass
    with step('sigmoid is finite at extreme inputs'):
        extreme = torch.tensor([-10000.0, -100.0, 0.0, 100.0, 10000.0])
        s = sigmoid(extreme)
        assert torch.isfinite(s).all()
        assert s[0] == 0.0 and s[-1] == 1.0
