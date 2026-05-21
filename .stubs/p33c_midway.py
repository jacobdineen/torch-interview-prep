"""
Problem 33c: midway cosine: progress=0.5 -> base/2

(Split from parent problem 33: Problem 28: Cosine Schedule With Linear Warmup)
"""
import math
import torch

def linear_warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
