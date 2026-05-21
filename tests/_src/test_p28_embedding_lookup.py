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
from p28_embedding_lookup import *

def test_p28_embedding_lookup():

    torch.manual_seed(0)
    V, D = 7, 4
    emb = MyEmbedding(V, D, padding_idx=0)
    with step("weight has shape (V, D)"):
        assert emb.weight.shape == (V, D)
    with step("padding_idx row is zero at init"):
        assert torch.equal(emb.weight[0], torch.zeros(D))
    with step("forward returns idx.shape + (D,)"):
        idx = torch.tensor([[1, 2, 0], [3, 4, 5]])
        out = emb(idx)
        assert out.shape == (2, 3, D)
    with step("padding position outputs zeros"):
        idx = torch.tensor([[1, 2, 0], [3, 4, 5]])
        out = emb(idx)
        assert torch.equal(out[0, 2], torch.zeros(D))
        assert torch.equal(out[0, 0], emb.weight[1])
    with step("gradient at padding_idx row is zero after backward"):
        idx2 = torch.tensor([0, 1, 2, 3])
        loss = emb(idx2).sum()
        if emb.weight.grad is not None: emb.weight.grad.zero_()
        loss.backward()
        assert torch.equal(emb.weight.grad[0], torch.zeros(D))
        assert not torch.equal(emb.weight.grad[1], torch.zeros(D))

