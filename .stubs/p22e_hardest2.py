"""
Problem 22e: hardest-triplet: positive loss when classes overlap

(Split from parent problem 22: Problem 58: Triplet Margin Loss)
"""
import torch

def hardest_triplet_loss(embeddings, labels, margin=1.0):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
