"""
Problem 08c: scatter_sum_rows sums values by row index (segment sum)

(Split from parent problem 08: Problem 08: Gather and Scatter)
"""
import torch

def scatter_sum_rows(values, index, num_rows):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
