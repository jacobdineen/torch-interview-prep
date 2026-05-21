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
from p03d_require_contiguous import *

def test_p03d_require_contiguous():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)
    pass
    pass
    pass
    with step('require_contiguous returns a contiguous tensor with same data'):
        t = x.transpose(0, 1)
        c = require_contiguous(t)
        assert c.is_contiguous()
        assert torch.equal(c, t.contiguous())
