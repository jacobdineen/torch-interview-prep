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
from p27a_train import *

def test_p27a_train():
    torch.manual_seed(0)
    p = 0.4
    d = MyDropout(p)
    x = torch.ones(10000)
    with step('train: drop rate is roughly p'):
        d.train()
        y = d(x)
        frac_zero = (y == 0).float().mean().item()
        assert abs(frac_zero - p) < 0.03, f'fraction zero was {frac_zero}, expected ~{p}'
    pass
    pass
    pass
