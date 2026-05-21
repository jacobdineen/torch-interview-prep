"""
Problem 19: Binary Cross-Entropy With Logits

Implement BCE-with-logits without calling F.binary_cross_entropy{_with_logits}.

  - bce_with_logits(logits, targets, pos_weight=None, reduction="mean")

For each element:
    L = -[ pos_weight * y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x)) ]

Use the numerically stable form:
    L = max(x, 0) - x * y + log(1 + exp(-|x|))            (pos_weight = 1)
With pos_weight w:
    L = (1 - y) * x + log1p(exp(-|x|))*(... )            -> derive carefully.

Test against F.binary_cross_entropy_with_logits.
"""

import torch
import torch.nn.functional as F

def bce_with_logits(logits, targets, pos_weight=None, reduction="mean"):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
