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
from p07d_masked_mean import *

def test_p07d_masked_mean():
    x = torch.tensor([-1.0, 0.0, 2.0, -3.0, 4.0])
    pass
    pass
    pass
    with step('masked_mean averages over true positions of the mask'):
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        assert torch.isclose(masked_mean(y, mask), torch.tensor(2.0))
