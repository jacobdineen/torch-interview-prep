"""
Step 0001: linear_beta_schedule

Part 1 — Noise Schedule
Build the variance (beta) schedule as a linear ramp over T diffusion steps.
"""
import torch  # noqa: F401


def linear_beta_schedule(T, beta_start=0.0001, beta_end=0.02):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
