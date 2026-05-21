"""
Problem 46e: chain: classic ImageNet stem 224 -> 112 -> 56 -> 56

(Split from parent problem 46: Problem 35: Conv Output Shape Calculation)
"""

def conv_chain_shape(h_in, layers):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
