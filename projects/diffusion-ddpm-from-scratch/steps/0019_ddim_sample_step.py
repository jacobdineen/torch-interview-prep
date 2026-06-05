"""
Step 0019: ddim_sample_step

Part 6 — DDIM Sampling
Take one deterministic (eta=0) DDIM reverse step from timestep t to t_prev.

t and t_prev are python ints. Predict eps, get x0_hat via predict_x0_from_noise. Let ab_prev = alpha_bar at t_prev, or 1 when t_prev<0. Return sqrt(ab_prev)*x0_hat + sqrt(1-ab_prev)*eps (shape (B,D)); deterministic, no added noise (eta=0).

Conventions: PyTorch tensors, batch-first. The noise schedule arrays (betas, alphas, alpha_bars)
have shape (T,); gather_at_timesteps indexes them by a (B,) timestep tensor and returns (B,1) so
the terms broadcast over the data. The model predicts the NOISE added to a sample (epsilon
parameterization); the training loss is MSE between predicted and true noise.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py diffusion-ddpm-from-scratch` (or the outline drawer) to see all signatures."""
import torch  # noqa: F401


def ddim_sample_step(params, x_t, t, t_prev, alpha_bars, time_dim):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
