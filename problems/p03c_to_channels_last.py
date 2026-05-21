"""
Problem 03c: to_channels_last permutes (B, C, H, W) -> (B, H, W, C)

(Split from parent problem 03: Problem 03: Reshape, View, Permute)
"""

import torch


def to_channels_last(x):
    print(x.shape)
    print(x.transpose(1, -1).shape)
    # 2,3,4,5 -> 2,4,5,3

    return x.permute(0, 2, 3, 1)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
