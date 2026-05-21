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
from p28e_gradient import *

def test_p28e_gradient():
    torch.manual_seed(0)
    V, D = (7, 4)
    emb = MyEmbedding(V, D, padding_idx=0)
    pass
    pass
    pass
    pass
    with step('gradient at padding_idx row is zero after backward'):
        idx2 = torch.tensor([0, 1, 2, 3])
        loss = emb(idx2).sum()
        if emb.weight.grad is not None:
            emb.weight.grad.zero_()
        loss.backward()
        assert torch.equal(emb.weight.grad[0], torch.zeros(D))
        assert not torch.equal(emb.weight.grad[1], torch.zeros(D))
