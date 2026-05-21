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
from p77e_causal_lm_loss import *

def test_p77e_causal_lm_loss():
    torch.manual_seed(0)
    V, D, L, H, FF, MAXT = (47, 16, 2, 4, 24, 16)
    model = MiniGPT(V, D, L, H, FF, MAXT, dropout=0.0)
    ids = torch.randint(0, V, (2, 8))
    logits = model(ids)
    pass
    pass
    pass
    pass
    with step('causal_lm_loss equals shifted cross-entropy'):
        expected = F.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
        assert torch.allclose(causal_lm_loss(logits, ids), expected, atol=1e-06)
    pass
