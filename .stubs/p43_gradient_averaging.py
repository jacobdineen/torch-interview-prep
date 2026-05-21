"""
Problem 65: Simulating Distributed Gradient Averaging

In DDP, after backward, gradients are all-reduced (averaged) across ranks before
the optimizer step. Implement the averaging step without using torch.distributed,
operating on a list of model replicas held on the same device.

  - average_gradients(models)
        * models: list of nn.Module that share the same parameter shapes.
        * For each matched parameter across replicas, replace each .grad with the
          mean of the .grads across replicas (in-place).
        * Skip parameters whose .grad is None on all replicas.
        * If grad is None on some but not all replicas, raise ValueError (this would
          be a real bug in a training step).

This pattern is what `torch.distributed.all_reduce(grad, op=ReduceOp.AVG)` does
under the hood.
"""

import torch
import torch.nn as nn


def average_gradients(models):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
