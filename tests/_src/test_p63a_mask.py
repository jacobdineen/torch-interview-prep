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
import math
import torch
from p63a_mask import *

def test_p63a_mask():
    T, W = (6, 3)
    mask = sliding_window_causal_mask(T, W)
    with step('mask shape and dtype'):
        assert mask.shape == (T, T) and mask.dtype == torch.bool
    pass
    torch.manual_seed(0)
    B, H, D = (2, 4, 8)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    pass
    pass
