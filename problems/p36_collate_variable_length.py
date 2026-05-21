"""
Problem 32: Custom Collate for Variable-Length Sequences

Sequences in a batch may have different lengths. Build a collate_fn that pads them
to the longest sequence in the batch and produces a mask of valid positions.

  - pad_collate(batch, pad_value=0)
        batch: list of (seq, label) where seq is a 1D LongTensor of variable length and
               label is a 0-D LongTensor.
        Returns:
            padded: (B, T_max)   LongTensor, with `pad_value` in padding positions.
            mask:   (B, T_max)   bool tensor, True for valid token positions.
            labels: (B,)         LongTensor.
"""

import torch
from torch.utils.data import DataLoader

def pad_collate(batch, pad_value=0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
