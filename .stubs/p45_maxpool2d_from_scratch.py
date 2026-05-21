"""
Problem 34: MaxPool2d From Scratch

Implement 2D max pooling without calling F.max_pool2d.

  - my_max_pool2d(x, kernel_size, stride=None, padding=0)
        x: (B, C, H, W)
        kernel_size: int (square)
        stride: int or None (defaults to kernel_size)
        padding: int (pads with -inf so padded positions never become the max)
        Returns: (B, C, Hout, Wout) max over each kernel window.

Hint: use F.unfold (or .unfold on the spatial dims) to extract windows.
"""

import torch
import torch.nn.functional as F

def my_max_pool2d(x, kernel_size, stride=None, padding=0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
