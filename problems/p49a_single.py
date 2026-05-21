"""
Problem 49a: single-step h_next and c_next match nn.LSTMCell

(Split from parent problem 49: Problem 38: LSTM Cell From Scratch)
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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
