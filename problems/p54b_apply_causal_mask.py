"""
Problem 54b: apply_causal_mask sets upper-triangle (strict) to -inf

(Split from parent problem 54: Problem 43: Causal Mask and Masked Attention)
"""

import math
import torch


def apply_causal_mask(scores):
    raise NotImplementedError

def causal_attention(q, k, v):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
