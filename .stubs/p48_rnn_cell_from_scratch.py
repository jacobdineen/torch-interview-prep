"""
Problem 37: Vanilla RNN Cell

Implement an Elman RNN cell:

    h_t = tanh(x_t @ W_ih.T + b_ih + h_{t-1} @ W_hh.T + b_hh)

  - VanillaRNNCell(input_size, hidden_size)
        Parameters:
            W_ih  (hidden_size, input_size)
            W_hh  (hidden_size, hidden_size)
            b_ih  (hidden_size,)
            b_hh  (hidden_size,)
        Use Kaiming-uniform or any reasonable init; tests reset weights anyway.
        forward(x, h_prev) where x is (B, input_size), h_prev is (B, hidden_size).
        Returns h_next of shape (B, hidden_size).

Also implement:
  - run_rnn(cell, X, h0)
        X:  (B, T, input_size)
        h0: (B, hidden_size)
        Returns (outputs, h_T) where outputs has shape (B, T, hidden_size) and
        h_T is the final hidden state.
"""

import torch
import torch.nn as nn

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        raise NotImplementedError

    def forward(self, x, h_prev):
        raise NotImplementedError

def run_rnn(cell, X, h0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
