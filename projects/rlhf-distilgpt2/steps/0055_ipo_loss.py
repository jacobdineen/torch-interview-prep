"""
Step 0055: ipo_loss

Part 7 — Preference Optimization Alternatives
IPO loss: (h - 1/(2*tau))^2 where h is the policy-minus-reference log-ratio gap.
"""
import torch  # noqa: F401


def ipo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, tau):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
