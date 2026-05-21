"""
Problem 50d: masked_mean_pool averages over real positions per sequence

(Split from parent problem 50: Problem 39: Padding, Masking, and Sequence Aggregation)
"""

import torch


def pad_and_mask(seqs):
    raise NotImplementedError

def masked_mean_pool(padded, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
