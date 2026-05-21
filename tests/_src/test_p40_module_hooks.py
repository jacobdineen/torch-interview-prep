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
import torch.nn as nn
from p40_module_hooks import *

def test_p40_module_hooks():

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    x = torch.randn(3, 4)
    names = ["0", "1"]
    acts = collect_activations(model, x, names)
    with step("returned dict has the requested layer names as keys"):
        assert set(acts.keys()) == set(names)
    with step("activation at layer 0 equals Linear(0)(x)"):
        assert torch.allclose(acts["0"], model[0](x), atol=1e-6)
    with step("activation at layer 1 equals ReLU(Linear(0)(x))"):
        assert torch.allclose(acts["1"], nn.functional.relu(model[0](x)), atol=1e-6)
    with step("forward hooks are removed after call (no leak)"):
        for _, mod in model.named_modules():
            assert len(mod._forward_hooks) == 0
    with step("repeated calls don't double-register hooks"):
        acts2 = collect_activations(model, x, names)
        assert torch.allclose(acts2["0"], model[0](x), atol=1e-6)
        for _, mod in model.named_modules():
            assert len(mod._forward_hooks) == 0
    with step("hooks are cleaned up even when the forward raises"):
        bad = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        class BadInput: pass
        raised = False
        try:
            collect_activations(bad, BadInput(), ["0"])
        except Exception:
            raised = True
        assert raised
        for _, mod in bad.named_modules():
            assert len(mod._forward_hooks) == 0

