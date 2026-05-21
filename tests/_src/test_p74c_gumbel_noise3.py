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
import math
import torch
from p74c_gumbel_noise3 import *

def test_p74c_gumbel_noise3():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    samples = gumbel_noise((100000,), generator=g)
    pass
    pass
    with step('gumbel_noise empirical variance ~ pi^2 / 6'):
        assert abs(samples.var().item() - math.pi ** 2 / 6) < 0.2
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pass
    pass
    pass
