"""
Step 0023: linear_warmup_schedule

Part 3 — SFT Training Loop
Linearly ramp the LR from 0 to base_lr over warmup_steps, then hold.
"""
import torch  # noqa: F401


def linear_warmup_schedule(step, warmup_steps, base_lr):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
