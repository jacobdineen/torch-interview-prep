import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import torch
from p03_reshape_view_permute import *

def test_p03_reshape_view_permute():

    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)
    with step("flatten_batch collapses all dims except batch"):
        f = flatten_batch(x)
        assert f.shape == (2, 60)
        assert torch.equal(f, x.reshape(2, -1))
    with step("swap_last_two transposes the last two dims"):
        y = torch.randn(2, 3, 4)
        s = swap_last_two(y)
        assert s.shape == (2, 4, 3)
        assert torch.equal(s, y.transpose(-1, -2))
    with step("to_channels_last permutes (B, C, H, W) -> (B, H, W, C)"):
        img = torch.randn(2, 3, 4, 5)
        cl = to_channels_last(img)
        assert cl.shape == (2, 4, 5, 3)
        assert torch.equal(cl[1, 2, 3, 0], img[1, 0, 2, 3])
    with step("require_contiguous returns a contiguous tensor with same data"):
        t = x.transpose(0, 1)
        c = require_contiguous(t)
        assert c.is_contiguous()
        assert torch.equal(c, t.contiguous())

