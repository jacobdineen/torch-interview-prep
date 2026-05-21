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
from p48a_single import *

def test_p48a_single():
    torch.manual_seed(0)
    B, T, I, H = (2, 5, 4, 6)
    cell = VanillaRNNCell(I, H)
    ref = nn.RNNCell(I, H, nonlinearity='tanh')
    with torch.no_grad():
        cell.W_ih.copy_(ref.weight_ih)
        cell.W_hh.copy_(ref.weight_hh)
        cell.b_ih.copy_(ref.bias_ih)
        cell.b_hh.copy_(ref.bias_hh)
    with step('single-step cell matches nn.RNNCell'):
        x = torch.randn(B, I)
        h = torch.randn(B, H)
        assert torch.allclose(cell(x, h), ref(x, h), atol=1e-05)
    X = torch.randn(B, T, I)
    h0 = torch.zeros(B, H)
    outputs, h_T = run_rnn(cell, X, h0)
    pass
    pass
