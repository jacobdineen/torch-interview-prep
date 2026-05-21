"""
Problem 51c: per-row outputs match running the LSTM on the unpadded sequence

(Split from parent problem 51: Problem 64: Bidirectional LSTM with Padded/Packed Sequences)
"""

import torch
import torch.nn as nn


def encode_packed(lstm, padded, lengths):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
