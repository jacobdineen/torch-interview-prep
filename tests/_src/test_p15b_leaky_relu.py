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
from p15b_leaky_relu import *

def test_p15b_leaky_relu():
    torch.manual_seed(0)
    x = torch.randn(50) * 5
    pass
    with step('leaky_relu matches F.leaky_relu(0.1)'):
        assert torch.allclose(leaky_relu(x, 0.1), torch.nn.functional.leaky_relu(x, 0.1))
    pass
    pass
    pass
    pass
