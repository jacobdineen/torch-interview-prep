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
from p77c_exactly import *

def test_p77c_exactly():
    torch.manual_seed(0)
    V, D, L, H, FF, MAXT = (47, 16, 2, 4, 24, 16)
    model = MiniGPT(V, D, L, H, FF, MAXT, dropout=0.0)
    ids = torch.randint(0, V, (2, 8))
    logits = model(ids)
    pass
    pass
    with step('exactly one Parameter of shape (V, D) — tied embedding/lm_head'):
        n_vd = sum((1 for p in model.parameters() if tuple(p.shape) == (V, D)))
        assert n_vd == 1, f'got {n_vd}'
    pass
    pass
    pass
