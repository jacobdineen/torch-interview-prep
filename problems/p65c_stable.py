"""
Problem 65c: stable at very large logits (online softmax)

(Split from parent problem 65: Problem 65: Tiled (Flash-Style) Attention)
"""

import math
import torch


def flash_attention_tiled(q, k, v, block_size_q=32, block_size_k=32):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
