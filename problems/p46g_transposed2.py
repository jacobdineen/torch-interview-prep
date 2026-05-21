"""
Problem 46g: transposed conv: 28 -> 56 with k=2 s=2 p=0

(Split from parent problem 46: Problem 35: Conv Output Shape Calculation)
"""


def transposed_conv_out_shape(h_in, kernel, stride=1, padding=0, output_padding=0, dilation=1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
