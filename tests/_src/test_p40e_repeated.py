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
from p40e_repeated import *

def test_p40e_repeated():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(3, 4)
    names = ['0', '1']
    acts = collect_activations(model, x, names)
    pass
    pass
    pass
    pass
    with step("repeated calls don't double-register hooks"):
        acts2 = collect_activations(model, x, names)
        assert torch.allclose(acts2['0'], model[0](x), atol=1e-06)
        for _, mod in model.named_modules():
            assert len(mod._forward_hooks) == 0
    pass
