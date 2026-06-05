"""
Step 0014: train_denoiser

Part 4 — Training Objective
Train the denoiser for n_steps on a fixed batch and return the list of per-step losses.
"""
import torch  # noqa: F401


def train_denoiser(params, x0, alpha_bars, T, time_dim, lr, n_steps, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
