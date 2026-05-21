"""
Problem 44a: my_conv2d matches F.conv2d (stride={...}, padding={...}, kernel={...})

(Split from parent problem 44: Problem 33: Conv2d From Scratch (using F.unfold))
"""

import torch
import torch.nn.functional as F


def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
