"""
Problem 63d: window=1: output equals v (each token sees only itself)

(Split from parent problem 63: Problem 63: Sliding Window Attention)
"""

import math
import torch


def sliding_window_causal_mask(T, window_size):
    raise NotImplementedError

def sliding_window_attention(q, k, v, window_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
