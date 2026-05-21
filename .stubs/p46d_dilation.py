"""
Problem 46d: dilation=2 effective kernel: 7 -> 2

(Split from parent problem 46: Problem 35: Conv Output Shape Calculation)
"""

def conv_out_shape(h_in, kernel, stride=1, padding=0, dilation=1):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
