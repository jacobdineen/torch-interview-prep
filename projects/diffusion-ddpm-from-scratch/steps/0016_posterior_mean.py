"""
Step 0016: posterior_mean

Part 5 — DDPM Sampling
Compute the mean of the DDPM posterior q(x_{t-1}|x_t,x0).

Gather beta_t, alpha_t, alpha_bar_t at t and alpha_bar_prev at t-1 (use alpha_bar_prev=1 where t==0). Return coef0*x0 + coeft*x_t, where coef0 = beta_t*sqrt(alpha_bar_prev)/(1-alpha_bar_t) and coeft = (1-alpha_bar_prev)*sqrt(alpha_t)/(1-alpha_bar_t); shape (B,D).

Conventions: PyTorch tensors, batch-first. The noise schedule arrays (betas, alphas, alpha_bars)
have shape (T,); gather_at_timesteps indexes them by a (B,) timestep tensor and returns (B,1) so
the terms broadcast over the data. The model predicts the NOISE added to a sample (epsilon
parameterization); the training loss is MSE between predicted and true noise.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py diffusion-ddpm-from-scratch` (or the outline drawer) to see all signatures."""
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
