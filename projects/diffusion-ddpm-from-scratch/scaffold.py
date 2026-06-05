"""Build a Diffusion Model (DDPM) from Scratch — end-to-end demo.

    python projects.py diffusion-ddpm-from-scratch --scaffold

Trains your denoiser on a toy 2-D distribution (two clusters), then samples from
pure noise with both DDPM and DDIM and reports how close the generated cloud is
to the data. Pure CPU, well under a minute.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    T, D, time_dim = 50, 2, 16
    # toy data: two gaussian blobs
    n = 512
    centers = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])
    x0 = centers[torch.randint(0, 2, (n,))] + 0.3 * torch.randn(n, D)

    betas = linear_beta_schedule(T)
    alphas = compute_alphas(betas)
    alpha_bars = compute_alpha_bars(alphas)
    params = init_denoiser_params(D, time_dim, [128, 128], seed=0)

    print(f"DDPM: T={T} data_dim={D} time_dim={time_dim}")
    hist = train_denoiser(params, x0, alpha_bars, T, time_dim, lr=2e-2, n_steps=5000, generator=g)
    print(f"loss: {hist[0]:.4f} -> {hist[-1]:.4f}")

    with torch.no_grad():
        s_ddpm = generate_samples(params, 512, D, betas, alphas, alpha_bars, time_dim, method="ddpm", generator=g)
        s_ddim = generate_samples(params, 512, D, betas, alphas, alpha_bars, time_dim, method="ddim", ddim_steps=20, generator=g)

    def near_data(s):  # fraction of samples within 1.0 of a center
        d = torch.minimum((s - centers[0]).norm(dim=1), (s - centers[1]).norm(dim=1))
        return float((d < 1.0).float().mean())
    print(f"DDPM samples near a mode: {near_data(s_ddpm):.2f}   DDIM: {near_data(s_ddim):.2f}")
    print("ok" if hist[-1] < hist[0] and near_data(s_ddpm) > 0.6 else "warning: weak fit")


if __name__ == "__main__":
    main()
