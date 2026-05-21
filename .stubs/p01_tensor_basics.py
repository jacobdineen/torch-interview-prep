"""
Problem 01: Tensor Basics

Create, inspect, and convert tensors.

Implement:
  - make_zeros(shape, dtype)
  - make_range(start, end, step) -> 1D tensor [start, start+step, ... < end]
  - to_dtype_device(x, dtype, device)
  - tensor_info(x) -> {"shape": tuple, "dtype": torch.dtype, "device": torch.device, "numel": int}
"""

import torch


def make_zeros(shape, dtype=torch.float32):
    raise NotImplementedError


def make_range(start, end, step=1):
    raise NotImplementedError


def to_dtype_device(x, dtype, device):
    raise NotImplementedError


def tensor_info(x):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
