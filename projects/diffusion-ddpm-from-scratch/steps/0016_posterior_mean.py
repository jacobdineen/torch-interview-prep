"""
Step 0016: posterior_mean

Part 5 — DDPM Sampling
Compute the mean of the DDPM posterior q(x_{t-1}|x_t,x0).
"""
import torch  # noqa: F401


def posterior_mean(x0, x_t, t, betas, alphas, alpha_bars):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
