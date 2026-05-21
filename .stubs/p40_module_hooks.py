"""
Problem 63: Forward Hooks to Collect Intermediate Activations

PyTorch lets you attach hooks to nn.Module instances so you can capture inputs and
outputs without modifying the model code. Implement a small utility that does this
robustly (with cleanup) for a list of named submodules.

  - collect_activations(model, x, layer_names)
        * `layer_names` is a list of strings naming submodules (as in
          `dict(model.named_modules())`).
        * Run the forward pass once on `x`.
        * Return a dict mapping each name to that submodule's output tensor.
        * Hooks must be removed before the function returns (use try/finally) so
          repeated calls don't leak.
"""

import torch
import torch.nn as nn


def collect_activations(model, x, layer_names):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
