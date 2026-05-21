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
from p77f_training import *

def test_p77f_training():
    torch.manual_seed(0)
    V, D, L, H, FF, MAXT = (47, 16, 2, 4, 24, 16)
    model = MiniGPT(V, D, L, H, FF, MAXT, dropout=0.0)
    ids = torch.randint(0, V, (2, 8))
    logits = model(ids)
    pass
    pass
    pass
    pass
    pass
    with step('training reduces loss by at least 50% over 100 steps'):
        opt = torch.optim.AdamW(model.parameters(), lr=0.003)
        batch_x = torch.randint(0, V, (16, 12))
        initial = causal_lm_loss(model(batch_x), batch_x).item()
        for _ in range(100):
            opt.zero_grad()
            loss = causal_lm_loss(model(batch_x), batch_x)
            loss.backward()
            opt.step()
        assert loss.item() < initial * 0.5
