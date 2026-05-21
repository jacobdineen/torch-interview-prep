"""
Problem 33: Conv2d From Scratch (using F.unfold)

Implement a 2D convolution forward pass equivalent to nn.Conv2d (no groups, no
dilation past 1 needed). Don't use F.conv2d.

  - my_conv2d(x, weight, bias=None, stride=1, padding=0)
        x:       (B, Cin, H, W)
        weight:  (Cout, Cin, kH, kW)
        bias:    (Cout,) or None
        stride:  int (square stride)
        padding: int (square padding)
        Returns: (B, Cout, Hout, Wout)

Hint: use torch.nn.functional.unfold to extract sliding patches, then matmul with the
flattened weight. Reshape back to (B, Cout, Hout, Wout).
"""

import torch
import torch.nn.functional as F

def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
