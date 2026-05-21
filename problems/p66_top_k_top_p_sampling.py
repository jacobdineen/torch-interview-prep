"""
Problem 48: Top-k and Top-p (Nucleus) Sampling

Implement the two filters used to truncate the next-token distribution before
sampling from a language model.

  - top_k_filter(logits, k)
        logits: (..., V)
        Keep the top-k logits in each row; set all others to -inf.
        Returns a Tensor of the same shape as logits.

  - top_p_filter(logits, p)
        logits: (..., V)
        Sort logits descending. Compute softmax. Accumulate probability; keep tokens
        until cumulative prob exceeds p. Always keep at least one token.
        Set everything else's logit to -inf.
        Returns a Tensor of the same shape as logits.

  - sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None)
        Apply temperature (divide logits), then top_k_filter then top_p_filter (if set),
        then sample from softmax via torch.multinomial.
        Returns a LongTensor of shape (...) (one less dim than logits, i.e., V is removed).
"""

import torch

def top_k_filter(logits, k):
    raise NotImplementedError

def top_p_filter(logits, p):
    raise NotImplementedError

def sample_token(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
