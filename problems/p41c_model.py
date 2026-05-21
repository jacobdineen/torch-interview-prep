"""
Problem 41c: model parameters restored bit-exact

(Split from parent problem 41: Problem 60: Save and Load a Training Checkpoint)
"""

import torch


def save_checkpoint(path, model, optimizer, epoch, best_metric):
    raise NotImplementedError

def load_checkpoint(path, model, optimizer):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
