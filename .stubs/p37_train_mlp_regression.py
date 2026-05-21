"""
Problem 30: End-to-End MLP Training

Put it together: a small MLP regressor trained on synthetic data with a real training
loop. You may use nn.Linear, nn.ReLU, torch.optim, etc.

Implement:
  - build_mlp(in_dim, hidden_dims, out_dim): returns an nn.Module (e.g., nn.Sequential)
        Architecture: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear (no final activation).
  - train_one_epoch(model, optimizer, x, y, batch_size=32): returns mean MSE loss across batches.
  - fit(model, x, y, epochs, lr, batch_size=32): trains and returns the final epoch's loss.
"""

import torch
import torch.nn as nn

def build_mlp(in_dim, hidden_dims, out_dim):
    raise NotImplementedError

def train_one_epoch(model, optimizer, x, y, batch_size=32):
    raise NotImplementedError

def fit(model, x, y, epochs, lr, batch_size=32):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
