"""Hidden reference implementations for diffusion-ddpm-from-scratch. Steps call earlier references
(shared namespace). One def per step with a one-line docstring."""
import math

import torch
import torch.nn.functional as F  # noqa: F401

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Build the variance (beta) schedule as a linear ramp over T diffusion steps.

    Return torch.linspace(beta_start, beta_end, T): a length-T tensor with betas[0]=beta_start and betas[-1]=beta_end, evenly spaced.
    """
    return torch.linspace(beta_start, beta_end, T)


def compute_alphas(betas):
    """Compute per-step alphas as one minus the betas."""
    return 1.0 - betas


def compute_alpha_bars(alphas):
    """Compute cumulative alpha products (alpha_bar) along the time axis."""
    return torch.cumprod(alphas, dim=0)


def gather_at_timesteps(buf, t):
    """Index a 1-D schedule buffer at the given timesteps and reshape to (B,1) for broadcasting."""
    return buf[t].reshape(-1, 1)

def sqrt_alpha_bar_terms(alpha_bars, t):
    """Return sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) gathered at timesteps t."""
    ab = gather_at_timesteps(alpha_bars, t)
    sqrt_ab = torch.sqrt(ab)
    sqrt_one_minus_ab = torch.sqrt(1.0 - ab)
    return sqrt_ab, sqrt_one_minus_ab


def q_sample(x0, t, noise, alpha_bars):
    """Apply the closed-form forward noising to produce x_t from x0 and noise at timesteps t."""
    sqrt_ab, sqrt_one_minus_ab = sqrt_alpha_bar_terms(alpha_bars, t)
    return sqrt_ab * x0 + sqrt_one_minus_ab * noise


def sample_timesteps_and_noise(x0, T, generator=None):
    """Draw random integer timesteps in [0, T) and standard-normal noise shaped like x0."""
    B = x0.shape[0]
    t = torch.randint(0, T, (B,), generator=generator, dtype=torch.long, device=x0.device)
    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=x0.device)
    return t, noise

def sinusoidal_time_embedding(t, dim):
    """Map integer timesteps to Transformer-style sinusoidal feature vectors.

    Let half=dim//2 and freqs=exp(-(arange(half)/max(half,1))*log(10000)). With args = t[:,None].float()*freqs (shape (B,half)), return a (B,dim) float32 tensor whose even columns (0::2) are sin(args) and odd columns (1::2) are cos(args) (so t=0 gives sin 0, cos 1).
    """
    device = t.device
    half = dim // 2
    t = t.float().unsqueeze(1)  # (B, 1)
    exponents = torch.arange(half, device=device, dtype=torch.float32) / max(half, 1)
    freqs = torch.exp(exponents * (-math.log(10000.0)))  # (half,)
    args = t * freqs.unsqueeze(0)  # (B, half)
    emb = torch.zeros(t.shape[0], dim, device=device, dtype=torch.float32)
    emb[:, 0::2] = torch.sin(args)
    emb[:, 1::2] = torch.cos(args[:, : dim - half])
    return emb


def init_denoiser_params(data_dim, time_dim, hidden, seed=0):
    """Initialize MLP weights/biases mapping data+time features to a data-dim output.

    Return a list of (W, b) tuples, one per layer, for sizes [data_dim+time_dim] + list(hidden) + [data_dim]. Each W has shape (in_dim, out_dim) initialized as randn*0.1; each b has shape (out_dim,) initialized to zeros; both float32 with requires_grad=True. Seed with torch.Generator().manual_seed(seed) for determinism.
    """
    g = torch.Generator().manual_seed(seed)
    sizes = [data_dim + time_dim] + list(hidden) + [data_dim]
    params = []
    for i in range(len(sizes) - 1):
        in_d, out_d = sizes[i], sizes[i + 1]
        W = (torch.randn(in_d, out_d, generator=g) * 0.1).requires_grad_(True)
        b = torch.zeros(out_d, requires_grad=True)
        params.append((W, b))
    return params


def denoiser_mlp_forward(params, h):
    """Run the MLP forward with ReLU after every layer except the linear last."""
    n = len(params)
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < n - 1:
            h = F.relu(h)
    return h


def predict_noise(params, x_t, t, time_dim):
    """Predict the noise added to x_t at timestep t using the time-conditioned denoiser."""
    time_emb = sinusoidal_time_embedding(t, time_dim)
    h = torch.cat([x_t, time_emb], dim=-1)
    return denoiser_mlp_forward(params, h)

def diffusion_loss(params, x0, alpha_bars, T, time_dim, generator=None):
    """Compute the DDPM training loss: MSE between the denoiser's predicted noise and the true noise."""
    t, noise = sample_timesteps_and_noise(x0, T, generator=generator)
    x_t = q_sample(x0, t, noise, alpha_bars)
    predicted_noise = predict_noise(params, x_t, t, time_dim)
    return F.mse_loss(predicted_noise, noise)


def denoiser_train_step(params, x0, alpha_bars, T, time_dim, lr, generator=None):
    """Run one SGD step on the diffusion loss and return the scalar loss value before the update."""
    for (W, b) in params:
        if W.grad is not None:
            W.grad.detach_()
            W.grad.zero_()
        if b.grad is not None:
            b.grad.detach_()
            b.grad.zero_()
    loss = diffusion_loss(params, x0, alpha_bars, T, time_dim, generator=generator)
    loss.backward()
    with torch.no_grad():
        for (W, b) in params:
            W -= lr * W.grad
            b -= lr * b.grad
    return float(loss.detach())


def train_denoiser(params, x0, alpha_bars, T, time_dim, lr, n_steps, generator=None):
    """Train the denoiser for n_steps on a fixed batch and return the list of per-step losses."""
    history = []
    for _ in range(n_steps):
        history.append(denoiser_train_step(params, x0, alpha_bars, T, time_dim, lr, generator=generator))
    return history

def predict_x0_from_noise(x_t, t, noise, alpha_bars):
    """Recover the predicted clean sample x0 from a noisy x_t and the noise."""
    sqrt_ab, sqrt_one_minus_ab = sqrt_alpha_bar_terms(alpha_bars, t)
    return (x_t - sqrt_one_minus_ab * noise) / sqrt_ab


def posterior_mean(x0, x_t, t, betas, alphas, alpha_bars):
    """Compute the mean of the DDPM posterior q(x_{t-1}|x_t,x0).

    Gather beta_t, alpha_t, alpha_bar_t at t and alpha_bar_prev at t-1 (use alpha_bar_prev=1 where t==0). Return coef0*x0 + coeft*x_t, where coef0 = beta_t*sqrt(alpha_bar_prev)/(1-alpha_bar_t) and coeft = (1-alpha_bar_prev)*sqrt(alpha_t)/(1-alpha_bar_t); shape (B,D).
    """
    beta_t = gather_at_timesteps(betas, t)
    alpha_t = gather_at_timesteps(alphas, t)
    alpha_bar_t = gather_at_timesteps(alpha_bars, t)
    t_prev = (t - 1).clamp(min=0)
    alpha_bar_prev = gather_at_timesteps(alpha_bars, t_prev)
    # alpha_bar_prev = 1 when t == 0
    alpha_bar_prev = torch.where((t == 0).reshape(-1, 1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)
    coef0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
    coeft = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
    return coef0 * x0 + coeft * x_t


def ddpm_sample_step(params, x_t, t, betas, alphas, alpha_bars, time_dim, generator=None):
    """Take one reverse DDPM denoising step from x_t to x_{t-1}.

    t is a python int; broadcast it to a (B,) tensor. Predict noise, recover x0_hat via predict_x0_from_noise, then mean=posterior_mean(...). If t==0 return mean. Else return mean + sqrt(posterior_variance)*z with z~randn(generator), posterior_variance = beta_t*(1-alpha_bar_{t-1})/(1-alpha_bar_t).
    """
    B = x_t.shape[0]
    t_batch = torch.full((B,), int(t), dtype=torch.long, device=x_t.device)
    noise = predict_noise(params, x_t, t_batch, time_dim)
    x0_hat = predict_x0_from_noise(x_t, t_batch, noise, alpha_bars)
    mean = posterior_mean(x0_hat, x_t, t_batch, betas, alphas, alpha_bars)
    if int(t) == 0:
        return mean
    beta_t = gather_at_timesteps(betas, t_batch)
    alpha_bar_t = gather_at_timesteps(alpha_bars, t_batch)
    t_prev = t_batch - 1
    alpha_bar_prev = gather_at_timesteps(alpha_bars, t_prev)
    posterior_variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    z = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
    return mean + torch.sqrt(posterior_variance) * z


def ddpm_sample_loop(params, shape, betas, alphas, alpha_bars, time_dim, generator=None):
    """Run the full reverse DDPM chain from pure noise x_T down to x_0."""
    T = betas.shape[0]
    x_t = torch.randn(shape, generator=generator, dtype=betas.dtype)
    for t in range(T - 1, -1, -1):
        x_t = ddpm_sample_step(params, x_t, t, betas, alphas, alpha_bars, time_dim, generator=generator)
    return x_t

def ddim_sample_step(params, x_t, t, t_prev, alpha_bars, time_dim):
    """Take one deterministic (eta=0) DDIM reverse step from timestep t to t_prev.

    t and t_prev are python ints. Predict eps, get x0_hat via predict_x0_from_noise. Let ab_prev = alpha_bar at t_prev, or 1 when t_prev<0. Return sqrt(ab_prev)*x0_hat + sqrt(1-ab_prev)*eps (shape (B,D)); deterministic, no added noise (eta=0).
    """
    t_batch = torch.full((x_t.shape[0],), t, dtype=torch.long, device=x_t.device)
    eps = predict_noise(params, x_t, t_batch, time_dim)
    x0_hat = predict_x0_from_noise(x_t, t_batch, eps, alpha_bars)
    if t_prev < 0:
        ab_prev = torch.ones(x_t.shape[0], 1, dtype=x_t.dtype, device=x_t.device)
    else:
        idx = torch.full((x_t.shape[0],), t_prev, dtype=torch.long, device=x_t.device)
        ab_prev = gather_at_timesteps(alpha_bars, idx)
    return torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps


def ddim_sample_loop(params, shape, alpha_bars, time_dim, n_steps):
    """Generate samples by iterating deterministic DDIM steps over evenly-spaced timesteps.

    Build ts = torch.linspace(T-1, 0, n_steps).round().long().tolist(); start x=torch.randn(shape). For each index i with timestep t, set t_prev = ts[i+1] if it exists else -1, and x = ddim_sample_step(params, x, t, t_prev, alpha_bars, time_dim). Return the final x of shape `shape`.
    """
    T = alpha_bars.shape[0]
    ts = torch.linspace(T - 1, 0, n_steps).round().long().tolist()
    x = torch.randn(shape)
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        x = ddim_sample_step(params, x, t, t_prev, alpha_bars, time_dim)
    return x


def generate_samples(params, n_samples, data_dim, betas, alphas, alpha_bars, time_dim, method="ddpm", ddim_steps=None, generator=None):
    """Generate samples from the trained denoiser using either DDPM or DDIM sampling."""
    shape = (n_samples, data_dim)
    if method == "ddim":
        steps = ddim_steps if ddim_steps is not None else alpha_bars.shape[0]
        return ddim_sample_loop(params, shape, alpha_bars, time_dim, steps)
    return ddpm_sample_loop(params, shape, betas, alphas, alpha_bars, time_dim, generator=generator)
