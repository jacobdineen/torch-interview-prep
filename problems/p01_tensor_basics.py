"""
Problem 01: Tensor Basics

Create, inspect, and convert tensors.

Implement:
  - make_zeros(shape, dtype)
  - make_range(start, end, step) -> 1D tensor [start, start+step, ... < end]
  - to_dtype_device(x, dtype, device)
  - tensor_info(x) -> {"shape": tuple, "dtype": torch.dtype, "device": torch.device, "numel": int}
"""

import os
import sys

import torch


def make_zeros(shape, dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype)


def make_range(start, end, step=1):
    return torch.tensor([i for i in range(start, end, step)])


def to_dtype_device(x, dtype, device):
    return torch.tensor(x, dtype=dtype, device=device)


def tensor_info(x):
    return {"shape": x.shape, "dtype": x.dtype, "device": x.device, "numel": x.numel()}


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
