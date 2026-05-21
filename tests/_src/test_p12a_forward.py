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
from p12a_forward import *

def test_p12a_forward():
    torch.manual_seed(0)
    with step('forward matches torch.nn.functional.softplus over a wide range'):
        x = torch.linspace(-50, 50, steps=11)
        y = stable_softplus(x)
        expected = torch.nn.functional.softplus(x)
        assert torch.allclose(y, expected, atol=1e-05)
    pass
    pass
