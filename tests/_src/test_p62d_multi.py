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
from p62d_multi import *

def test_p62d_multi():
    torch.manual_seed(0)
    B, T, D, H = (2, 6, 16, 4)
    mha = CachedMHA(D, H)
    mha.eval()
    x = torch.randn(B, T, D)
    full_out, full_cache = mha(x, cache=None)
    pass
    pass
    pass
    with step('multi-token-chunk streaming also matches the full pass'):
        cache2 = None
        o1, cache2 = mha(x[:, :2], cache=cache2)
        o2, cache2 = mha(x[:, 2:5], cache=cache2)
        o3, cache2 = mha(x[:, 5:], cache=cache2)
        chunked = torch.cat([o1, o2, o3], dim=1)
        assert torch.allclose(chunked, full_out, atol=1e-05)
