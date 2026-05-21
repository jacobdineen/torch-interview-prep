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
from p05b_per_batch_max import *

def test_p05b_per_batch_max():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 16, 16)
    pass
    with step('per_batch_max takes the max over all non-batch dims'):
        y = torch.randn(4, 5, 6)
        pbm = per_batch_max(y)
        assert pbm.shape == (4,)
        assert torch.allclose(pbm, y.flatten(1).max(dim=1).values)
    pass
    pass
