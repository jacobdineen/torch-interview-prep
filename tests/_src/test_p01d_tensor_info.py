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
from p01d_tensor_info import *

def test_p01d_tensor_info():
    pass
    pass
    pass
    with step('tensor_info returns shape, dtype, device, numel'):
        info = tensor_info(torch.randn(3, 4))
        assert tuple(info['shape']) == (3, 4)
        assert info['dtype'] == torch.float32
        assert info['numel'] == 12
