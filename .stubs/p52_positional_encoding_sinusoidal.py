"""
Problem 40: Sinusoidal Positional Encoding (Vaswani et al.)

Implement the fixed sinusoidal PE used in the original Transformer:

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))

  - sinusoidal_positional_encoding(seq_len, d_model)
        Returns a (seq_len, d_model) FloatTensor.
        d_model must be even.

  - add_positional_encoding(x)
        x: (B, T, d_model)
        Returns x + PE (PE generated on x's device/dtype).
"""

import math
import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    raise NotImplementedError

def add_positional_encoding(x):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
