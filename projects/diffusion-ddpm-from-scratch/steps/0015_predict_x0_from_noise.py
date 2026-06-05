"""
Step 0015: predict_x0_from_noise

Part 5 — DDPM Sampling
Recover the predicted clean sample x0 from a noisy x_t and the noise.
"""
import torch  # noqa: F401


def predict_x0_from_noise(x_t, t, noise, alpha_bars):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
