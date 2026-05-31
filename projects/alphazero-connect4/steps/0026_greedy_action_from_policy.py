"""
Step 0026: greedy_action_from_policy

Part 3 — Action Masking and Policy Sampling
The legal column with the highest logit.
"""
import torch  # noqa: F401


def greedy_action_from_policy(logits, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
