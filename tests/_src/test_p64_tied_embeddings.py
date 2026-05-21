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
import torch.nn.functional as F
from p64_tied_embeddings import *

def test_p64_tied_embeddings():

    torch.manual_seed(0)
    V, D = 100, 8
    lm = TiedLM(V, D)
    ids = torch.randint(0, V, (3, 5))
    out = lm(ids)
    with step("logits shape (B, T, V)"):
        assert out.shape == (3, 5, V)
    with step("logits equal embed(ids) @ embed.weight.T"):
        expected = lm.embed(ids) @ lm.embed.weight.t()
        assert torch.allclose(out, expected, atol=1e-6)
    with step("exactly one Parameter (the embedding) — lm_head is tied"):
        n_unique = len({id(p) for p in lm.parameters()})
        assert n_unique == 1
    with step("backward populates embed.weight.grad"):
        targets = torch.randint(0, V, (3, 5))
        loss = F.cross_entropy(out.view(-1, V), targets.view(-1))
        loss.backward()
        assert lm.embed.weight.grad is not None
        assert torch.isfinite(lm.embed.weight.grad).all()
    with step("trainable parameter count equals V * D (no separate head)"):
        n_params = sum(p.numel() for p in lm.parameters() if p.requires_grad)
        assert n_params == V * D

