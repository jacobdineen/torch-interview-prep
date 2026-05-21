"""
Problem 66a: top_k_filter keeps top-k, masks the rest to -inf

(Split from parent problem 66: Problem 48: Top-k and Top-p (Nucleus) Sampling)
"""

import torch


def top_k_filter(logits, k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
