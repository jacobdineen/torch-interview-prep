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
from p39c_autocast3 import *

def test_p39c_autocast3():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(4, 8)
    out = forward_autocast(model, x, dtype=torch.bfloat16, device_type='cpu')
    pass
    pass
    with step('autocast output is numerically close to fp32 forward'):
        out_f32 = model(x)
        assert (out.float() - out_f32).abs().max().item() < 0.5
    pass
    pass
