"""
Problem 01d: tensor_info returns shape, dtype, device, numel

(Split from parent problem 01: Problem 01: Tensor Basics)
"""
import os
import sys
import torch

def tensor_info(x):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
