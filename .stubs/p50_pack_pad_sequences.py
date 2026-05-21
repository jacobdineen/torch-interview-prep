"""
Problem 39: Padding, Masking, and Sequence Aggregation

You're given a list of token-embedding sequences of variable length. Pad them into a
tensor and compute masked summaries.

  - pad_and_mask(seqs):
        seqs: list of (T_i, D) float tensors (same D, varying T_i).
        Returns:
            padded: (B, T_max, D)
            mask:   (B, T_max) bool, True for real positions.
            lengths:(B,) Long, original lengths.

  - masked_mean_pool(padded, mask):
        Mean over real positions per sequence -> (B, D).

  - last_real_state(padded, lengths):
        Return (B, D) with padded[b, lengths[b] - 1] for each b.
"""

import torch

def pad_and_mask(seqs):
    raise NotImplementedError

def masked_mean_pool(padded, mask):
    raise NotImplementedError

def last_real_state(padded, lengths):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
