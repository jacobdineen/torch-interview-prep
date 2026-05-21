"""
Problem 66d: sample_token returns indices within the top-k filter

(Split from parent problem 66: Problem 48: Top-k and Top-p (Nucleus) Sampling)
"""

import torch


def sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
