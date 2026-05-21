"""
Problem 20: Focal Loss (multi-class)

Focal loss down-weights well-classified examples. For multi-class with integer
targets:

    p_t  = softmax(logits)[:, target]            # probability of the true class
    L    = - alpha * (1 - p_t)^gamma * log(p_t)

where alpha can be a per-class weight tensor of shape (C,) (or None / scalar).

Implement:
  - focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean")

Compute log_softmax once (in a numerically stable way) and reuse it.
"""

import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
