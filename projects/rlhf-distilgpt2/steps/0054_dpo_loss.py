"""
Step 0054: dpo_loss

Part 7 — Preference Optimization Alternatives
DPO loss: -log sigmoid(beta * (policy_logratio - reference_logratio)).
"""
import torch  # noqa: F401


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
