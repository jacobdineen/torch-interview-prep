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
from p03b_swap_last_two import *

def test_p03b_swap_last_two():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)
    pass
    with step('swap_last_two transposes the last two dims'):
        y = torch.randn(2, 3, 4)
        s = swap_last_two(y)
        assert s.shape == (2, 4, 3)
        assert torch.equal(s, y.transpose(-1, -2))
    pass
    pass
