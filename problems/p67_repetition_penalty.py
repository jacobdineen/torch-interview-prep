"""
Problem 67: Repetition Penalty (CTRL-style)

A common decoder trick to discourage the model from repeating itself: for each
previously-generated token, divide its logit by `penalty` if positive, multiply by
`penalty` if negative. (So in both cases the effective probability decreases.)

Implement:

  - apply_repetition_penalty(logits, generated_ids, penalty=1.0)
        logits: (B, V) -- next-token logits for B sequences.
        generated_ids: (B, T) long -- the tokens generated so far for each row.
        For each (b, t), let v = logits[b, generated_ids[b, t]].
        If v > 0: divide by penalty.
        If v <= 0: multiply by penalty.
        Returns the modified logits (a NEW tensor, same shape).

  - penalize_unique(logits, generated_ids, penalty=1.0)
        Same as above but each token is penalized only once even if it appears
        multiple times in `generated_ids`.

Setting penalty=1.0 must be a no-op for both.
"""

import torch


def apply_repetition_penalty(logits, generated_ids, penalty=1.0):
    raise NotImplementedError


def penalize_unique(logits, generated_ids, penalty=1.0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
