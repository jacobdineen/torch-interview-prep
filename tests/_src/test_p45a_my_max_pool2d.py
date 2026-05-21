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
import torch.nn.functional as F
from p45a_my_max_pool2d import *

def test_p45a_my_max_pool2d():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 10, 10)
    for k, s, p in [(2, 2, 0), (3, 1, 1), (3, 2, 1)]:
        with step(f'my_max_pool2d matches F.max_pool2d (k={k}, s={s}, p={p})'):
            out = my_max_pool2d(x, kernel_size=k, stride=s, padding=p)
            expected = F.max_pool2d(x, kernel_size=k, stride=s, padding=p)
            assert out.shape == expected.shape
            assert torch.allclose(out, expected, atol=1e-06)
    pass
