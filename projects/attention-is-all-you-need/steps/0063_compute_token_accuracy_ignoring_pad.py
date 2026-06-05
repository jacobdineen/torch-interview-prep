"""
Step 0063: compute_token_accuracy_ignoring_pad

Part 8 — Training Objective and Schedule
Compute the fraction of non-pad positions where the predicted argmax token matches the gold token.
"""
import torch  # noqa: F401


def compute_token_accuracy_ignoring_pad(logits_or_logprobs, gold, pad_id=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
