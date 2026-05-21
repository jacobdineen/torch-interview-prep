"""
Problem 38: LSTM Cell From Scratch

Implement an LSTM cell matching torch.nn.LSTMCell:

    gates = x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh   # shape (B, 4*H)
    Split into i, f, g, o (in that order, as PyTorch does).
    i = sigmoid(i);  f = sigmoid(f);  o = sigmoid(o);  g = tanh(g)
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)

  - LSTMCellFromScratch(input_size, hidden_size)
        Parameters:
            W_ih (4H, I), W_hh (4H, H), b_ih (4H,), b_hh (4H,)
        forward(x, (h_prev, c_prev)) -> (h_next, c_next)
"""

import torch
import torch.nn as nn

class LSTMCellFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, state):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
