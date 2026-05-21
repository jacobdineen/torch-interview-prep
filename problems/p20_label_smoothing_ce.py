"""
Problem 47: Label Smoothing Cross-Entropy

Standard cross-entropy uses one-hot targets. Label smoothing replaces the one-hot
target with a soft distribution:

    target_one_hot * (1 - smoothing) + smoothing / num_classes

  - label_smoothing_ce(logits, targets, smoothing=0.1, reduction="mean")
        logits:  (N, C)
        targets: (N,) long, in [0, C)
        Returns the loss as a 0-D Tensor (or (N,) Tensor if reduction="none").

You should:
    log_probs = log_softmax(logits, dim=-1)             # (N, C)  -- stable
    nll      = -log_probs.gather(1, targets[:, None]).squeeze(1)   # (N,)
    smooth_loss = -log_probs.mean(dim=-1)                # (N,)
    loss = (1 - smoothing) * nll + smoothing * smooth_loss
    reduce as requested.

Sanity properties:
  - smoothing=0  ==  cross-entropy
  - smoothing=1  yields -mean(log_probs) regardless of targets
"""

import torch
import torch.nn.functional as F

def label_smoothing_ce(logits, targets, smoothing=0.1, reduction="mean"):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
