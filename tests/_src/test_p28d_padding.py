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
from p28d_padding import *

def test_p28d_padding():
    torch.manual_seed(0)
    V, D = (7, 4)
    emb = MyEmbedding(V, D, padding_idx=0)
    pass
    pass
    pass
    with step('padding position outputs zeros'):
        idx = torch.tensor([[1, 2, 0], [3, 4, 5]])
        out = emb(idx)
        assert torch.equal(out[0, 2], torch.zeros(D))
        assert torch.equal(out[0, 0], emb.weight[1])
    pass
