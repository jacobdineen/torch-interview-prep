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
from p15a_relu import *

def test_p15a_relu():
    torch.manual_seed(0)
    x = torch.randn(50) * 5
    with step('relu matches torch.relu'):
        assert torch.allclose(relu(x), torch.relu(x))
    pass
    pass
    pass
    pass
    pass
