"""
Problem 77d: causal: perturbing future tokens leaves earlier logits unchanged

(Split from parent problem 77: Problem 77: MiniGPT End-to-End (Capstone))
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):

    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.0):
        raise NotImplementedError

    def forward(self, input_ids):
        raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
