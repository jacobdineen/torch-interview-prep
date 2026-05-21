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
from p16a_logsumexp import *

def test_p16a_logsumexp():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    with step('logsumexp matches torch.logsumexp'):
        assert torch.allclose(logsumexp(x, dim=1), torch.logsumexp(x, dim=1), atol=1e-06)
    pass
    pass
    pass
    pass
    pass
