"""
Problem 29c: kaiming_normal_ fan_out,linear: std ~ 1/sqrt(fan_out)

(Split from parent problem 29: Problem 25: Weight Initialization)
"""

import math
import torch


def kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu"):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
