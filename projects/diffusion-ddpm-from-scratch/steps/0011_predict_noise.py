"""
Step 0011: predict_noise

Part 3 — Time Embedding & Denoiser
Predict the noise added to x_t at timestep t using the time-conditioned denoiser.
"""
import torch  # noqa: F401


def predict_noise(params, x_t, t, time_dim):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
