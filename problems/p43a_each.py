"""
Problem 43a: each replica's grad equals the mean across replicas

(Split from parent problem 43: Problem 65: Simulating Distributed Gradient Averaging)
"""

import torch
import torch.nn as nn


def average_gradients(models):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
