"""
Step 0002: compute_alphas

Part 1 — Noise Schedule
Compute per-step alphas as one minus the betas.
"""
import torch  # noqa: F401


def compute_alphas(betas):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
