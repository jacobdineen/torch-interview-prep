"""
Problem 35: Conv Output Shape Calculation

A common interview warmup: given input size and conv hyperparameters, return the
output spatial size.

  - conv_out_shape(h_in, kernel, stride=1, padding=0, dilation=1)
        Returns the integer floor((h_in + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1

  - conv_chain_shape(h_in, layers)
        layers: list of dicts, each with keys among {kernel, stride, padding, dilation}.
        Returns the spatial size after applying them in order.

  - transposed_conv_out_shape(h_in, kernel, stride=1, padding=0, output_padding=0, dilation=1)
        Returns (h_in - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
"""

def conv_out_shape(h_in, kernel, stride=1, padding=0, dilation=1):
    raise NotImplementedError

def conv_chain_shape(h_in, layers):
    raise NotImplementedError

def transposed_conv_out_shape(h_in, kernel, stride=1, padding=0, output_padding=0, dilation=1):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
