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
from p64c_exactly import *

def test_p64c_exactly():
    torch.manual_seed(0)
    V, D = (100, 8)
    lm = TiedLM(V, D)
    ids = torch.randint(0, V, (3, 5))
    out = lm(ids)
    pass
    pass
    with step('exactly one Parameter (the embedding) — lm_head is tied'):
        n_unique = len({id(p) for p in lm.parameters()})
        assert n_unique == 1
    pass
    pass
