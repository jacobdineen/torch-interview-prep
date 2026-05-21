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
from p62a_full import *

def test_p62a_full():
    torch.manual_seed(0)
    B, T, D, H = (2, 6, 16, 4)
    mha = CachedMHA(D, H)
    mha.eval()
    x = torch.randn(B, T, D)
    full_out, full_cache = mha(x, cache=None)
    with step('full-pass output shape (B, T, D)'):
        assert full_out.shape == (B, T, D)
    pass
    pass
    pass
