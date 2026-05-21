"""
Problem 40b: activation at layer 0 equals Linear(0)(x)

(Split from parent problem 40: Problem 63: Forward Hooks to Collect Intermediate Activations)
"""
import torch
import torch.nn as nn

def collect_activations(model, x, layer_names):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
