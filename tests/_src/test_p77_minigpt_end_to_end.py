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
from p77_minigpt_end_to_end import *

def test_p77_minigpt_end_to_end():

    torch.manual_seed(0)
    V, D, L, H, FF, MAXT = 47, 16, 2, 4, 24, 16
    model = MiniGPT(V, D, L, H, FF, MAXT, dropout=0.0)
    ids = torch.randint(0, V, (2, 8))
    logits = model(ids)
    with step("logits shape (B, T, V)"):
        assert logits.shape == (2, 8, V)
    with step("logits require grad"):
        assert logits.requires_grad
    with step("exactly one Parameter of shape (V, D) — tied embedding/lm_head"):
        n_vd = sum(1 for p in model.parameters() if tuple(p.shape) == (V, D))
        assert n_vd == 1, f"got {n_vd}"
    with step("causal: perturbing future tokens leaves earlier logits unchanged"):
        ids2 = ids.clone(); ids2[:, 4:] = torch.randint(0, V, ids2[:, 4:].shape)
        logits2 = model(ids2)
        assert torch.allclose(logits[:, :4], logits2[:, :4], atol=1e-5)
    with step("causal_lm_loss equals shifted cross-entropy"):
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        assert torch.allclose(causal_lm_loss(logits, ids), expected, atol=1e-6)
    with step("training reduces loss by at least 50% over 100 steps"):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        batch_x = torch.randint(0, V, (16, 12))
        initial = causal_lm_loss(model(batch_x), batch_x).item()
        for _ in range(100):
            opt.zero_grad()
            loss = causal_lm_loss(model(batch_x), batch_x)
            loss.backward(); opt.step()
        assert loss.item() < initial * 0.5

