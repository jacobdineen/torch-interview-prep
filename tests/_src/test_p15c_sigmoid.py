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
from p15c_sigmoid import *

def test_p15c_sigmoid():
    torch.manual_seed(0)
    x = torch.randn(50) * 5
    pass
    pass
    with step('sigmoid matches torch.sigmoid'):
        assert torch.allclose(sigmoid(x), torch.sigmoid(x), atol=1e-06)
    pass
    pass
    pass
