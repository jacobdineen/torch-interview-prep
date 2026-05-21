"""
Problem 17: MSE and Huber Loss From Scratch

Implement (no torch.nn.functional.mse_loss / smooth_l1_loss / huber_loss):

  - mse_loss(pred, target, reduction="mean")
        reduction in {"mean", "sum", "none"}.

  - huber_loss(pred, target, delta=1.0, reduction="mean")
        L_huber(r) =  0.5 * r^2                if |r| <= delta
                      delta * (|r| - 0.5*delta) otherwise
        where r = pred - target.
"""

import torch

def mse_loss(pred, target, reduction="mean"):
    raise NotImplementedError

def huber_loss(pred, target, delta=1.0, reduction="mean"):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
