"""
Problem 52b: PE[0] is sin(0)=0 at even cols and cos(0)=1 at odd cols

(Split from parent problem 52: Problem 40: Sinusoidal Positional Encoding (Vaswani et al.))
"""
import math
import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
