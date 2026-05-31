"""
Step 0057: orpo_loss

Part 7 — Preference Optimization Alternatives
ORPO (reference-free): chosen NLL plus an odds-ratio preference term.
Expects average (length-normalized) log-probs, both < 0.
"""
import torch  # noqa: F401


def orpo_loss(policy_chosen_logps, policy_rejected_logps, beta):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
