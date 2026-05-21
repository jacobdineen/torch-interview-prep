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
from p62b_cache import *

def test_p62b_cache():
    torch.manual_seed(0)
    B, T, D, H = (2, 6, 16, 4)
    mha = CachedMHA(D, H)
    mha.eval()
    x = torch.randn(B, T, D)
    full_out, full_cache = mha(x, cache=None)
    pass
    with step('cache k, v shape (B, H, T, head_dim)'):
        assert full_cache['k'].shape == (B, H, T, D // H)
        assert full_cache['v'].shape == (B, H, T, D // H)
    pass
    pass
