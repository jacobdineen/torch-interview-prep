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
from p16d_softmax2 import *

def test_p16d_softmax2():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    pass
    pass
    pass
    with step('softmax rows sum to 1'):
        s = softmax(x, dim=1)
        assert torch.allclose(s.sum(dim=1), torch.ones(4), atol=1e-06)
    pass
    pass
