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
from p48b_run_rnn import *

def test_p48b_run_rnn():
    torch.manual_seed(0)
    B, T, I, H = (2, 5, 4, 6)
    cell = VanillaRNNCell(I, H)
    ref = nn.RNNCell(I, H, nonlinearity='tanh')
    with torch.no_grad():
        cell.W_ih.copy_(ref.weight_ih)
        cell.W_hh.copy_(ref.weight_hh)
        cell.b_ih.copy_(ref.bias_ih)
        cell.b_hh.copy_(ref.bias_hh)
    pass
    X = torch.randn(B, T, I)
    h0 = torch.zeros(B, H)
    outputs, h_T = run_rnn(cell, X, h0)
    with step('run_rnn outputs have shape (B, T, H) and final state (B, H)'):
        assert outputs.shape == (B, T, H)
        assert h_T.shape == (B, H)
    pass
