"""
Step 0053: dpo_ref_logratios

Part 7 — Preference Optimization Alternatives
Reference model's log-ratio: ref_chosen - ref_rejected.
"""
import torch  # noqa: F401


def dpo_ref_logratios(ref_chosen_logps, ref_rejected_logps):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
