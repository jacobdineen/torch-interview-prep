"""
Step 0012: diffusion_loss

Part 4 — Training Objective
Compute the DDPM training loss: MSE between the denoiser's predicted noise and the true noise.
"""
import torch  # noqa: F401


def diffusion_loss(params, x0, alpha_bars, T, time_dim, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
