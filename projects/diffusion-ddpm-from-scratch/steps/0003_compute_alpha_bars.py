"""
Step 0003: compute_alpha_bars

Part 1 — Noise Schedule
Compute cumulative alpha products (alpha_bar) along the time axis.
"""
import torch  # noqa: F401


def compute_alpha_bars(alphas):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
