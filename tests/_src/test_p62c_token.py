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
from p62c_token import *

def test_p62c_token():
    torch.manual_seed(0)
    B, T, D, H = (2, 6, 16, 4)
    mha = CachedMHA(D, H)
    mha.eval()
    x = torch.randn(B, T, D)
    full_out, full_cache = mha(x, cache=None)
    pass
    pass
    with step('token-by-token streaming matches the full-pass output'):
        cache = None
        outs = []
        for t in range(T):
            o, cache = mha(x[:, t:t + 1], cache=cache)
            outs.append(o)
        streamed = torch.cat(outs, dim=1)
        assert torch.allclose(streamed, full_out, atol=1e-05)
    pass
