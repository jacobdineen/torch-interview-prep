"""
Problem 38a: accumulated gradient equals mean-over-full-batch gradient

(Split from parent problem 38: Problem 61: Gradient Accumulation)
"""
import torch

def train_step_with_accumulation(model, optimizer, micro_batches, loss_fn, accum_steps):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
