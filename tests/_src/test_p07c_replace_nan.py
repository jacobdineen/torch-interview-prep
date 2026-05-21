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
from p07c_replace_nan import *

def test_p07c_replace_nan():
    x = torch.tensor([-1.0, 0.0, 2.0, -3.0, 4.0])
    pass
    pass
    with step('replace_nan substitutes a value for NaN entries'):
        nx = torch.tensor([1.0, float('nan'), 3.0, float('nan')])
        out = replace_nan(nx, -1.0)
        assert torch.equal(out, torch.tensor([1.0, -1.0, 3.0, -1.0]))
    pass
