"""
Problem 13b: stop_gradient_at: forward equals x; grads blocked above threshold

(Split from parent problem 13: Problem 13: Controlling Gradient Flow)
"""

import torch
import torch.nn as nn


def stop_gradient_at(x, threshold):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
