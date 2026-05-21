"""
Problem 18: Cross-Entropy From Scratch

Implement multi-class cross-entropy from logits and integer targets, WITHOUT using
F.cross_entropy, F.log_softmax, F.nll_loss.

  - cross_entropy(logits, targets, reduction="mean", ignore_index=None)
        logits  : (N, C)
        targets : (N,)  long; values in [0, C) unless equal to ignore_index
        ignore_index: if given, those positions are excluded from the loss and the
                      "mean" reduction divides by the number of non-ignored examples.
        Compute the stable log-softmax yourself.

Hint: use the max-shift trick for stability, then NLL = -log_softmax[target].
"""

import torch
import torch.nn.functional as F

def cross_entropy(logits, targets, reduction="mean", ignore_index=None):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
