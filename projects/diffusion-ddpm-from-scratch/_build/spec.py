"""Build spec for the diffusion-ddpm-from-scratch project."""

TITLE = 'Build a Diffusion Model (DDPM) from Scratch'

PARTS = [
    ("Noise Schedule",
     "Build the linear beta schedule and the alpha / cumulative-alpha buffers that define the forward diffusion."),
    ("Forward Diffusion Process",
     "Noise a clean sample to an arbitrary timestep in one shot, and draw random timesteps + noise for training."),
    ("Time Embedding & Denoiser",
     "Sinusoidal timestep embeddings and a small MLP that predicts the noise added to a noisy sample."),
    ("Training Objective",
     "The simple noise-prediction MSE loss, a single optimization step, and the training loop."),
    ("DDPM Sampling",
     "Reverse the diffusion one stochastic step at a time, from pure noise back to a sample."),
    ("DDIM Sampling",
     "Deterministic accelerated sampling and an end-to-end generation helper."),
]

STEPS = [
    # Part 1 - Noise Schedule
    ("linear_beta_schedule", 0), ("compute_alphas", 0),
    ("compute_alpha_bars", 0), ("gather_at_timesteps", 0),
    # Part 2 - Forward Diffusion
    ("sqrt_alpha_bar_terms", 1), ("q_sample", 1), ("sample_timesteps_and_noise", 1),
    # Part 3 - Time Embedding & Denoiser
    ("sinusoidal_time_embedding", 2), ("init_denoiser_params", 2),
    ("denoiser_mlp_forward", 2), ("predict_noise", 2),
    # Part 4 - Training Objective
    ("diffusion_loss", 3), ("denoiser_train_step", 3), ("train_denoiser", 3),
    # Part 5 - DDPM Sampling
    ("predict_x0_from_noise", 4), ("posterior_mean", 4),
    ("ddpm_sample_step", 4), ("ddpm_sample_loop", 4),
    # Part 6 - DDIM Sampling
    ("ddim_sample_step", 5), ("ddim_sample_loop", 5), ("generate_samples", 5),
]


def step_id(i):
    return f"{i:04d}"
