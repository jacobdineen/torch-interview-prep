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
import copy
import torch
import torch.nn as nn
from p42c_ema_wrapper2 import *

def test_p42c_ema_wrapper2():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    ema = nn.Linear(4, 2)
    with torch.no_grad():
        ema.weight.zero_()
        ema.bias.zero_()
    init_w = model.weight.detach().clone()
    ema_update(ema, model, decay=0.9)
    pass
    m2 = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    w = EMAWrapper(m2, decay=0.5)
    sd_init = copy.deepcopy(w.ema_state_dict())
    pass
    with step('EMAWrapper: EMA params have requires_grad=False'):
        if hasattr(w, 'ema_model'):
            for p in w.ema_model.parameters():
                assert not p.requires_grad
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(1.0)
    sd_before = copy.deepcopy(m2.state_dict())
    w.update()
    sd_after = copy.deepcopy(w.ema_state_dict())
    pass
