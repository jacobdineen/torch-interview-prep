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
from p28b_padding_idx import *

def test_p28b_padding_idx():
    torch.manual_seed(0)
    V, D = (7, 4)
    emb = MyEmbedding(V, D, padding_idx=0)
    pass
    with step('padding_idx row is zero at init'):
        assert torch.equal(emb.weight[0], torch.zeros(D))
    pass
    pass
    pass
