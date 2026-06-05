"""
Step 0021: generate_samples

Part 6 — DDIM Sampling
Generate samples from the trained denoiser using either DDPM or DDIM sampling.
"""
import torch  # noqa: F401


def generate_samples(params, n_samples, data_dim, betas, alphas, alpha_bars, time_dim, method='ddpm', ddim_steps=None, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
