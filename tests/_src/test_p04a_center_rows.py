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
from p04a_center_rows import *

def test_p04a_center_rows():
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    with step('center_rows subtracts per-row mean'):
        c = center_rows(X)
        assert torch.allclose(c.mean(dim=1), torch.zeros(5), atol=1e-06)
    pass
    pass
    pass
