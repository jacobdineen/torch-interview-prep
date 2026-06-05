"""
Step 0017: ddpm_sample_step

Part 5 — DDPM Sampling
Take one reverse DDPM denoising step from x_t to x_{t-1}.

t is a python int; broadcast it to a (B,) tensor. Predict noise, recover x0_hat via predict_x0_from_noise, then mean=posterior_mean(...). If t==0 return mean. Else return mean + sqrt(posterior_variance)*z with z~randn(generator), posterior_variance = beta_t*(1-alpha_bar_{t-1})/(1-alpha_bar_t).

Conventions: PyTorch tensors, batch-first. The noise schedule arrays (betas, alphas, alpha_bars)
have shape (T,); gather_at_timesteps indexes them by a (B,) timestep tensor and returns (B,1) so
the terms broadcast over the data. The model predicts the NOISE added to a sample (epsilon
parameterization); the training loss is MSE between predicted and true noise.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py diffusion-ddpm-from-scratch` (or the outline drawer) to see all signatures."""
import torch  # noqa: F401


def ddpm_sample_step(params, x_t, t, betas, alphas, alpha_bars, time_dim, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
