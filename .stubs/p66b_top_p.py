"""
Problem 66b: top_p with p=0.9 keeps the top 2 (cumprob ~0.87)

(Split from parent problem 66: Problem 48: Top-k and Top-p (Nucleus) Sampling)
"""
import torch

def top_p_filter(logits, p):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
