"""
Problem 06b: stack_batch creates a new leading dim

(Split from parent problem 06: Problem 06: Concatenation, Stacking, Splitting)
"""
import torch

def stack_batch(items):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
