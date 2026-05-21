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
from p50d_masked_mean_pool import *

def test_p50d_masked_mean_pool():
    torch.manual_seed(0)
    seqs = [torch.randn(3, 5), torch.randn(1, 5), torch.randn(4, 5)]
    padded, mask, lengths = pad_and_mask(seqs)
    pass
    pass
    pass
    with step('masked_mean_pool averages over real positions per sequence'):
        mp = masked_mean_pool(padded, mask)
        assert mp.shape == (3, 5)
        assert torch.allclose(mp[0], seqs[0].mean(dim=0), atol=1e-06)
        assert torch.allclose(mp[1], seqs[1].mean(dim=0), atol=1e-06)
    pass
