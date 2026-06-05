"""
Step 0010: denoiser_mlp_forward

Part 3 — Time Embedding & Denoiser
Run the MLP forward with ReLU after every layer except the linear last.

Conventions: PyTorch tensors, batch-first. The noise schedule arrays (betas, alphas, alpha_bars)
have shape (T,); gather_at_timesteps indexes them by a (B,) timestep tensor and returns (B,1) so
the terms broadcast over the data. The model predicts the NOISE added to a sample (epsilon
parameterization); the training loss is MSE between predicted and true noise.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py diffusion-ddpm-from-scratch` (or the outline drawer) to see all signatures."""
import torch  # noqa: F401


def denoiser_mlp_forward(params, h):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
