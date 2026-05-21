"""
Problem 34a: large max_norm: gradients unchanged; returns pre-clip norm

(Split from parent problem 34: Problem 29: Gradient Clipping by Global Norm)
"""

import torch
import torch.nn as nn


def clip_grad_norm_(params, max_norm, eps=1e-6):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
