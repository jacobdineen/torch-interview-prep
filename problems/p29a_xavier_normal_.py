"""
Problem 29a: xavier_normal_ std ~ sqrt(2/(fan_in+fan_out))

(Split from parent problem 29: Problem 25: Weight Initialization)
"""

import math
import torch


def xavier_normal_(tensor, gain=1.0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
