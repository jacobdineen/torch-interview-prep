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
from p49b_multi import *

def test_p49b_multi():
    torch.manual_seed(0)
    B, I, H = (3, 4, 6)
    mine = LSTMCellFromScratch(I, H)
    ref = nn.LSTMCell(I, H)
    with torch.no_grad():
        mine.W_ih.copy_(ref.weight_ih)
        mine.W_hh.copy_(ref.weight_hh)
        mine.b_ih.copy_(ref.bias_ih)
        mine.b_hh.copy_(ref.bias_hh)
    pass
    with step('multi-step rollout matches nn.LSTMCell'):
        T = 4
        X = torch.randn(B, T, I)
        h_t, c_t = (torch.zeros(B, H), torch.zeros(B, H))
        h_ref_t, c_ref_t = (torch.zeros(B, H), torch.zeros(B, H))
        for t in range(T):
            h_t, c_t = mine(X[:, t], (h_t, c_t))
            h_ref_t, c_ref_t = ref(X[:, t], (h_ref_t, c_ref_t))
        assert torch.allclose(h_t, h_ref_t, atol=1e-05)
        assert torch.allclose(c_t, c_ref_t, atol=1e-05)
