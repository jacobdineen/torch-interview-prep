"""
Step 0018: ddpm_sample_loop

Part 5 — DDPM Sampling
Run the full reverse DDPM chain from pure noise x_T down to x_0.
"""
import torch  # noqa: F401


def ddpm_sample_loop(params, shape, betas, alphas, alpha_bars, time_dim, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
