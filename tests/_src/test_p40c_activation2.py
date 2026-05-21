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
from p40c_activation2 import *

def test_p40c_activation2():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(3, 4)
    names = ['0', '1']
    acts = collect_activations(model, x, names)
    pass
    pass
    with step('activation at layer 1 equals ReLU(Linear(0)(x))'):
        assert torch.allclose(acts['1'], nn.functional.relu(model[0](x)), atol=1e-06)
    pass
    pass
    pass
