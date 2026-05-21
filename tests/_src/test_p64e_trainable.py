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
from p64e_trainable import *

def test_p64e_trainable():
    torch.manual_seed(0)
    V, D = (100, 8)
    lm = TiedLM(V, D)
    ids = torch.randint(0, V, (3, 5))
    out = lm(ids)
    pass
    pass
    pass
    pass
    with step('trainable parameter count equals V * D (no separate head)'):
        n_params = sum((p.numel() for p in lm.parameters() if p.requires_grad))
        assert n_params == V * D
