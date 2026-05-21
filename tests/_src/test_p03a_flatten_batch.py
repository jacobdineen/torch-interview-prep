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
from p03a_flatten_batch import *

def test_p03a_flatten_batch():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)
    with step('flatten_batch collapses all dims except batch'):
        f = flatten_batch(x)
        assert f.shape == (2, 60)
        assert torch.equal(f, x.reshape(2, -1))
    pass
    pass
    pass
