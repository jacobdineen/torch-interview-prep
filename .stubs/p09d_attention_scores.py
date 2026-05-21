"""
Problem 09d: attention_scores: q @ k^T over the head_dim

(Split from parent problem 09: Problem 09: einsum Practice)
"""
import torch

def attention_scores(q, k):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
