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
from p74e_tau import *

def test_p74e_tau():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    samples = gumbel_noise((100000,), generator=g)
    pass
    pass
    pass
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pass
    with step('tau -> 0 makes the sample near one-hot'):
        g3 = torch.Generator().manual_seed(0)
        y_hot = gumbel_softmax(logits, tau=0.01, hard=False, generator=g3)
        assert y_hot.max(dim=-1).values.item() > 0.99
    pass
