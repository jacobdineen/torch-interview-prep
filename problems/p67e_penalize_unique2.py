"""
Problem 67e: penalize_unique applies once per token

(Split from parent problem 67: Problem 67: Repetition Penalty (CTRL-style))
"""

import torch


def penalize_unique(logits, generated_ids, penalty=1.0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
