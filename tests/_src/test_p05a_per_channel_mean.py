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
from p05a_per_channel_mean import *

def test_p05a_per_channel_mean():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 16, 16)
    with step('per_channel_mean averages over batch and spatial dims'):
        pcm = per_channel_mean(x)
        assert pcm.shape == (3,)
        assert torch.allclose(pcm, x.mean(dim=(0, 2, 3)), atol=1e-06)
    pass
    pass
    pass
