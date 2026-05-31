"""
Step 0024: masked_log_softmax

Part 3 — Action Masking and Policy Sampling
Log-softmax over legal actions only (illegal -> -inf).
"""
import torch  # noqa: F401


def masked_log_softmax(logits, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
