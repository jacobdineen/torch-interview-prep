"""
Step 0023: masked_policy_logits

Part 3 — Action Masking and Policy Sampling
Set logits of illegal actions (mask==0) to -inf.
"""
import torch  # noqa: F401


def masked_policy_logits(logits, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
