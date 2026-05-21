"""
Problem 53: SwiGLU FFN

The SwiGLU feed-forward block (LLaMA-style, replaces the standard MLP/GELU):

    SwiGLU(x) = (silu(x @ W_gate)) * (x @ W_up)
    out       = SwiGLU(x) @ W_down

where silu(z) = z * sigmoid(z). All weights are without bias.

  - SwiGLU(d_model, d_ff)
        Parameters:
            w_gate : Linear(d_model, d_ff, bias=False)
            w_up   : Linear(d_model, d_ff, bias=False)
            w_down : Linear(d_ff, d_model, bias=False)
        forward(x): x is (B, T, d_model); returns (B, T, d_model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
