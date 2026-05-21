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
from p16c_log_softmax import *

def test_p16c_log_softmax():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    pass
    pass
    with step('log_softmax matches torch.log_softmax'):
        assert torch.allclose(log_softmax(x, dim=-1), torch.log_softmax(x, dim=-1), atol=1e-06)
    pass
    pass
    pass
