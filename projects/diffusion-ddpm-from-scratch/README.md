# Build a Diffusion Model (DDPM) from Scratch

Implement a denoising diffusion probabilistic model end to end in PyTorch — the
β/α noise schedule, the closed-form forward noising, a sinusoidal-time-conditioned
denoiser that predicts the added noise, the simple MSE training objective, and
both stochastic **DDPM** and deterministic **DDIM** reverse sampling. On toy 2-D
data you watch pure noise turn back into the data distribution. **21 steps, 6 parts.**

| Part | Focus |
|------|-------|
| 1 | Noise schedule (β, α, ᾱ, gather-at-t) |
| 2 | Forward diffusion `q_sample`, timestep/noise sampling |
| 3 | Sinusoidal time embedding + the denoiser MLP |
| 4 | The noise-prediction MSE loss + training loop |
| 5 | DDPM reverse sampling (posterior mean + variance) |
| 6 | DDIM deterministic sampling + a generate helper |

```bash
uv run python projects.py diffusion-ddpm-from-scratch            # parts + steps
uv run python projects.py diffusion-ddpm-from-scratch --next     # next unsolved step
uv run python projects.py diffusion-ddpm-from-scratch --scaffold # train + sample demo
```
