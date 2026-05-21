"""
Problem 13a: param_norm matches the L2 norm of all params, no grad

(Split from parent problem 13: Problem 13: Controlling Gradient Flow)
"""

import torch
import torch.nn as nn


def param_norm(model):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
