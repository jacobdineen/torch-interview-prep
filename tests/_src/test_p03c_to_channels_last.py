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
from p03c_to_channels_last import *

def test_p03c_to_channels_last():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)
    pass
    pass
    with step('to_channels_last permutes (B, C, H, W) -> (B, H, W, C)'):
        img = torch.randn(2, 3, 4, 5)
        cl = to_channels_last(img)
        assert cl.shape == (2, 4, 5, 3)
        assert torch.equal(cl[1, 2, 3, 0], img[1, 0, 2, 3])
    pass
