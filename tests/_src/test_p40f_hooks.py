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
from p40f_hooks import *

def test_p40f_hooks():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(3, 4)
    names = ['0', '1']
    acts = collect_activations(model, x, names)
    pass
    pass
    pass
    pass
    pass
    with step('hooks are cleaned up even when the forward raises'):
        bad = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        class BadInput:
            pass
        raised = False
        try:
            collect_activations(bad, BadInput(), ['0'])
        except Exception:
            raised = True
        assert raised
        for _, mod in bad.named_modules():
            assert len(mod._forward_hooks) == 0
