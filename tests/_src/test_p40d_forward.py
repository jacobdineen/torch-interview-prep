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
from p40d_forward import *

def test_p40d_forward():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(3, 4)
    names = ['0', '1']
    acts = collect_activations(model, x, names)
    pass
    pass
    pass
    with step('forward hooks are removed after call (no leak)'):
        for _, mod in model.named_modules():
            assert len(mod._forward_hooks) == 0
    pass
    pass
