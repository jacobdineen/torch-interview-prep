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
import torch.nn as nn
from p39e_master import *

def test_p39e_master():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(4, 8)
    out = forward_autocast(model, x, dtype=torch.bfloat16, device_type='cpu')
    pass
    pass
    pass
    pass
    with step('master parameters remain float32 after AMP training'):
        for p in model.parameters():
            assert p.dtype == torch.float32
