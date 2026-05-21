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
from p01c_to_dtype_device import *

def test_p01c_to_dtype_device():
    pass
    pass
    with step('to_dtype_device converts dtype and device'):
        x = torch.zeros(2, 2, dtype=torch.float32)
        y = to_dtype_device(x, torch.int64, torch.device('cpu'))
        assert y.dtype == torch.int64
        assert y.device.type == 'cpu'
    pass
