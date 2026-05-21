"""
Problem 48c: run_rnn matches a manual time-step loop

(Split from parent problem 48: Problem 37: Vanilla RNN Cell)
"""
import torch
import torch.nn as nn

class VanillaRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        raise NotImplementedError

    def forward(self, x, h_prev):
        raise NotImplementedError

def run_rnn(cell, X, h0):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
