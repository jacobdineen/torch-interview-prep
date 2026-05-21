"""
Problem 54: ALiBi Positional Bias

Attention with Linear Biases adds a head-specific position-dependent bias to the
attention scores instead of using positional embeddings:

    score(q_i, k_j) += -m_h * |i - j|    (or simply -m_h * (i - j) for causal)

where m_h = 2 ^ (-8 * h / num_heads)  (for h = 1 .. num_heads).

Implement:

  - alibi_slopes(num_heads)
        Returns a 1-D FloatTensor of length num_heads of slopes m_h.

  - alibi_bias(num_heads, seq_len, causal=True)
        Returns a tensor of shape (num_heads, seq_len, seq_len). For causal=True the
        bias is added to all positions but causal masking is applied elsewhere — so
        the bias for (i, j) with j > i can be any finite value (typically also
        -m_h * (j - i), which is naturally negative). For causal=False, the bias is
        symmetric: -m_h * |i - j|.
"""

import torch


def alibi_slopes(num_heads):
    raise NotImplementedError


def alibi_bias(num_heads, seq_len, causal=True):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
