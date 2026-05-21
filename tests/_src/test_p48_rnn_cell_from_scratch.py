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
from p48_rnn_cell_from_scratch import *

def test_p48_rnn_cell_from_scratch():

    torch.manual_seed(0)
    B, T, I, H = 2, 5, 4, 6
    cell = VanillaRNNCell(I, H); ref = nn.RNNCell(I, H, nonlinearity="tanh")
    with torch.no_grad():
        cell.W_ih.copy_(ref.weight_ih); cell.W_hh.copy_(ref.weight_hh)
        cell.b_ih.copy_(ref.bias_ih); cell.b_hh.copy_(ref.bias_hh)
    with step("single-step cell matches nn.RNNCell"):
        x = torch.randn(B, I); h = torch.randn(B, H)
        assert torch.allclose(cell(x, h), ref(x, h), atol=1e-5)
    X = torch.randn(B, T, I); h0 = torch.zeros(B, H)
    outputs, h_T = run_rnn(cell, X, h0)
    with step("run_rnn outputs have shape (B, T, H) and final state (B, H)"):
        assert outputs.shape == (B, T, H)
        assert h_T.shape == (B, H)
    with step("run_rnn matches a manual time-step loop"):
        h_manual = h0
        outs_manual = []
        for t in range(T):
            h_manual = cell(X[:, t], h_manual)
            outs_manual.append(h_manual)
        outs_manual = torch.stack(outs_manual, dim=1)
        assert torch.allclose(outputs, outs_manual, atol=1e-6)
        assert torch.allclose(h_T, h_manual, atol=1e-6)

