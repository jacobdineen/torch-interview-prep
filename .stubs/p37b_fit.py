"""
Problem 37b: fit reduces loss by at least 20% over 10 epochs

(Split from parent problem 37: Problem 30: End-to-End MLP Training)
"""
import torch
import torch.nn as nn

def build_mlp(in_dim, hidden_dims, out_dim):
    raise NotImplementedError

def fit(model, x, y, epochs, lr, batch_size=32):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
