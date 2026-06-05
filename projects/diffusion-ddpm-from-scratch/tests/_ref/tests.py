"""Hidden tests for diffusion-ddpm-from-scratch. One test_<id>_<name>(ns) per step; deterministic, independent oracles."""
import math

import torch
import torch.nn.functional as F  # noqa: F401

def test_0001_linear_beta_schedule(ns):
    f = ns["linear_beta_schedule"]
    T = 20
    betas = f(T, beta_start=1e-4, beta_end=0.02)
    assert betas.shape == (T,)
    # Endpoints exact, interior values linearly spaced.
    assert torch.allclose(betas[0], torch.tensor(1e-4), atol=1e-7)
    assert torch.allclose(betas[-1], torch.tensor(0.02), atol=1e-7)
    expected = torch.tensor(
        [1e-4 + (0.02 - 1e-4) * i / (T - 1) for i in range(T)], dtype=betas.dtype
    )
    assert torch.allclose(betas, expected, atol=1e-5)
    # Monotonically increasing.
    assert torch.all(betas[1:] >= betas[:-1])


def test_0002_compute_alphas(ns):
    f = ns["compute_alphas"]
    betas = torch.tensor([0.0, 0.1, 0.25, 0.5, 1.0])
    alphas = f(betas)
    assert alphas.shape == betas.shape
    expected = torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0])
    assert torch.allclose(alphas, expected, atol=1e-6)


def test_0003_compute_alpha_bars(ns):
    f = ns["compute_alpha_bars"]
    alphas = torch.tensor([0.9, 0.8, 0.5, 1.0, 0.5])
    alpha_bars = f(alphas)
    assert alpha_bars.shape == alphas.shape
    # Manual running product.
    expected = []
    running = 1.0
    for a in [0.9, 0.8, 0.5, 1.0, 0.5]:
        running *= a
        expected.append(running)
    expected = torch.tensor(expected)
    assert torch.allclose(alpha_bars, expected, atol=1e-6)
    # With realistic schedule, alpha_bar is strictly decreasing in (0,1].
    betas = torch.linspace(1e-4, 0.02, 20)
    ab = f(1.0 - betas)
    assert torch.all(ab[1:] < ab[:-1])
    assert torch.all(ab > 0.0) and torch.all(ab <= 1.0)


def test_0004_gather_at_timesteps(ns):
    f = ns["gather_at_timesteps"]
    buf = torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0])
    t = torch.tensor([0, 2, 4, 1], dtype=torch.long)
    out = f(buf, t)
    assert out.shape == (4, 1)
    expected = torch.tensor([[10.0], [12.0], [14.0], [11.0]])
    assert torch.allclose(out, expected, atol=1e-6)
    # Broadcasts over a feature dim of size D.
    x = torch.zeros(4, 2)
    broadcasted = x + out
    assert broadcasted.shape == (4, 2)
    assert torch.allclose(broadcasted[:, 0], broadcasted[:, 1], atol=1e-6)

def test_0005_sqrt_alpha_bar_terms(ns):
    f = ns["sqrt_alpha_bar_terms"]
    T = 20
    betas = torch.linspace(1e-4, 0.02, T)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    t = torch.tensor([0, 5, 19, 12, 3], dtype=torch.long)
    sqrt_ab, sqrt_one_minus_ab = f(alpha_bars, t)
    assert sqrt_ab.shape == (5, 1)
    assert sqrt_one_minus_ab.shape == (5, 1)
    exp_ab = alpha_bars[t].reshape(-1, 1)
    assert torch.allclose(sqrt_ab, torch.sqrt(exp_ab), atol=1e-5)
    assert torch.allclose(sqrt_one_minus_ab, torch.sqrt(1.0 - exp_ab), atol=1e-5)
    # identity: sum of squares == 1
    assert torch.allclose(sqrt_ab ** 2 + sqrt_one_minus_ab ** 2,
                          torch.ones(5, 1), atol=1e-5)


def test_0006_q_sample(ns):
    f = ns["q_sample"]
    T = 20
    D = 2
    B = 16
    betas = torch.linspace(1e-4, 0.02, T)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    torch.manual_seed(0)
    x0 = torch.randn(B, D)
    noise = torch.randn(B, D)

    # t = 0: alpha_bar_0 ~= 1, so x_t ~= x0 (noise contribution ~ tiny)
    t0 = torch.zeros(B, dtype=torch.long)
    xt0 = f(x0, t0, noise, alpha_bars)
    assert xt0.shape == (B, D)
    sab0 = math.sqrt(float(alpha_bars[0]))
    somab0 = math.sqrt(1.0 - float(alpha_bars[0]))
    assert torch.allclose(xt0, sab0 * x0 + somab0 * noise, atol=1e-5)
    # at t=0 x_t is x0 plus a tiny noise term (~sqrt(beta_0)) -> close but not exact
    assert torch.allclose(xt0, x0, atol=6e-2)

    # general t: manual oracle per-row
    t = torch.tensor([0, 5, 19, 12, 3, 7, 1, 18, 9, 11, 2, 4, 6, 8, 10, 15],
                     dtype=torch.long)
    xt = f(x0, t, noise, alpha_bars)
    assert xt.shape == (B, D)
    assert torch.isfinite(xt).all()
    expected = torch.empty(B, D)
    for i in range(B):
        ab = float(alpha_bars[t[i]])
        expected[i] = math.sqrt(ab) * x0[i] + math.sqrt(1.0 - ab) * noise[i]
    assert torch.allclose(xt, expected, atol=1e-5)


def test_0007_sample_timesteps_and_noise(ns):
    f = ns["sample_timesteps_and_noise"]
    T = 20
    D = 2
    B = 16
    x0 = torch.zeros(B, D)

    g = torch.Generator().manual_seed(123)
    t, noise = f(x0, T, generator=g)
    assert t.shape == (B,)
    assert t.dtype == torch.long
    assert noise.shape == (B, D)
    assert noise.dtype == x0.dtype
    assert int(t.min()) >= 0 and int(t.max()) < T
    assert torch.isfinite(noise).all()

    # determinism: same seed -> same draws
    g2 = torch.Generator().manual_seed(123)
    t2, noise2 = f(x0, T, generator=g2)
    assert torch.equal(t, t2)
    assert torch.allclose(noise, noise2, atol=1e-7)

    # different seed -> generally different
    g3 = torch.Generator().manual_seed(999)
    t3, noise3 = f(x0, T, generator=g3)
    assert not torch.allclose(noise, noise3, atol=1e-3)

    # noise is roughly standard normal over a larger sample
    big = torch.zeros(20000, D)
    gb = torch.Generator().manual_seed(7)
    _, nb = f(big, T, generator=gb)
    assert abs(float(nb.mean())) < 0.05
    assert abs(float(nb.std()) - 1.0) < 0.05

def test_0008_sinusoidal_time_embedding(ns):
    import torch, math
    f = ns["sinusoidal_time_embedding"]
    B, dim = 5, 8
    t = torch.tensor([0, 1, 2, 7, 19], dtype=torch.long)
    emb = f(t, dim)
    # shape and dtype
    assert emb.shape == (B, dim)
    assert emb.dtype == torch.float32
    # bounded in [-1, 1]
    assert torch.all(emb <= 1.0 + 1e-6) and torch.all(emb >= -1.0 - 1e-6)
    assert torch.isfinite(emb).all()
    # finiteness/values: independent recompute of geometric-freq sin/cos
    half = dim // 2
    exps = torch.arange(half, dtype=torch.float32) / max(half, 1)
    freqs = torch.exp(exps * (-math.log(10000.0)))
    for bi in range(B):
        args = float(t[bi]) * freqs  # (half,)
        for j in range(half):
            assert abs(float(emb[bi, 2 * j]) - math.sin(float(args[j]))) < 1e-5
            assert abs(float(emb[bi, 2 * j + 1]) - math.cos(float(args[j]))) < 1e-5
    # t=0 -> sin=0, cos=1 pattern
    assert torch.allclose(emb[0, 0::2], torch.zeros(half), atol=1e-6)
    assert torch.allclose(emb[0, 1::2], torch.ones(half), atol=1e-6)


def test_0009_init_denoiser_params(ns):
    import torch
    f = ns["init_denoiser_params"]
    data_dim, time_dim, hidden = 2, 8, [64, 64]
    params = f(data_dim, time_dim, hidden, seed=0)
    sizes = [data_dim + time_dim] + hidden + [data_dim]
    # correct number of layers
    assert len(params) == len(sizes) - 1
    for i, (W, b) in enumerate(params):
        assert W.shape == (sizes[i], sizes[i + 1])
        assert b.shape == (sizes[i + 1],)
        assert W.requires_grad and b.requires_grad
        assert W.dtype == torch.float32 and b.dtype == torch.float32
        # biases start at zero
        assert torch.allclose(b, torch.zeros_like(b))
        # weights small (scaled ~0.1), nonzero, finite
        assert torch.isfinite(W).all()
        assert W.abs().max() < 1.0
        assert W.abs().sum() > 0
    # determinism w.r.t seed
    params2 = f(data_dim, time_dim, hidden, seed=0)
    assert torch.allclose(params[0][0], params2[0][0])
    # different seed -> different weights
    params3 = f(data_dim, time_dim, hidden, seed=1)
    assert not torch.allclose(params[0][0], params3[0][0])


def test_0010_denoiser_mlp_forward(ns):
    import torch, torch.nn.functional as F
    f = ns["denoiser_mlp_forward"]
    # build explicit tiny params: 3 -> 4 -> 2
    torch.manual_seed(0)
    W0 = torch.randn(3, 4); b0 = torch.randn(4)
    W1 = torch.randn(4, 2); b1 = torch.randn(2)
    params = [(W0, b0), (W1, b1)]
    h = torch.randn(6, 3)
    out = f(params, h)
    assert out.shape == (6, 2)
    # independent manual forward: relu on hidden, linear last
    expected = F.relu(h @ W0 + b0) @ W1 + b1
    assert torch.allclose(out, expected, atol=1e-5)
    # single layer (no hidden) is purely linear
    Wl = torch.randn(3, 2); bl = torch.randn(2)
    out1 = f([(Wl, bl)], h)
    assert torch.allclose(out1, h @ Wl + bl, atol=1e-5)


def test_0011_predict_noise(ns):
    import torch
    f = ns["predict_noise"]
    emb_f = ns["sinusoidal_time_embedding"]
    mlp_f = ns["denoiser_mlp_forward"]
    init_f = ns["init_denoiser_params"]
    data_dim, time_dim = 2, 8
    B = 16
    params = init_f(data_dim, time_dim, [64, 64], seed=0)
    torch.manual_seed(0)
    x_t = torch.randn(B, data_dim)
    t = torch.randint(0, 20, (B,))
    out = f(params, x_t, t, time_dim)
    # output shape is (B, data_dim)
    assert out.shape == (B, data_dim)
    assert torch.isfinite(out).all()
    # independent recompose: concat(x_t, emb) -> mlp
    emb = emb_f(t, time_dim)
    h = torch.cat([x_t, emb], dim=-1)
    expected = mlp_f(params, h)
    assert torch.allclose(out, expected, atol=1e-5)
    # determinism
    out2 = f(params, x_t, t, time_dim)
    assert torch.allclose(out, out2, atol=1e-7)

def test_0012_diffusion_loss(ns):
    import torch
    import torch.nn.functional as F
    diffusion_loss = ns["diffusion_loss"]
    init_denoiser_params = ns["init_denoiser_params"]
    linear_beta_schedule = ns["linear_beta_schedule"]
    compute_alphas = ns["compute_alphas"]
    compute_alpha_bars = ns["compute_alpha_bars"]
    sample_timesteps_and_noise = ns["sample_timesteps_and_noise"]
    q_sample = ns["q_sample"]
    predict_noise = ns["predict_noise"]

    T, D, B, time_dim = 20, 2, 16, 8
    betas = linear_beta_schedule(T)
    alpha_bars = compute_alpha_bars(compute_alphas(betas))
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)
    torch.manual_seed(0)
    x0 = torch.randn(B, D)

    # Loss must be a finite scalar.
    g = torch.Generator().manual_seed(123)
    loss = diffusion_loss(params, x0, alpha_bars, T, time_dim, generator=g)
    assert loss.shape == torch.Size([]), f"expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), "loss not finite"
    assert loss.item() >= 0.0, "MSE must be non-negative"

    # Independent oracle: replay the same generator and recompute MSE by hand.
    g2 = torch.Generator().manual_seed(123)
    t, noise = sample_timesteps_and_noise(x0, T, generator=g2)
    x_t = q_sample(x0, t, noise, alpha_bars)
    pred = predict_noise(params, x_t, t, time_dim)
    expected = ((pred - noise) ** 2).mean()
    assert torch.allclose(loss, expected, atol=1e-5), f"{loss.item()} vs {expected.item()}"

    # Determinism: same generator seed -> same loss.
    ga = torch.Generator().manual_seed(7)
    gb = torch.Generator().manual_seed(7)
    la = diffusion_loss(params, x0, alpha_bars, T, time_dim, generator=ga)
    lb = diffusion_loss(params, x0, alpha_bars, T, time_dim, generator=gb)
    assert torch.allclose(la, lb, atol=1e-6), "loss not deterministic under fixed generator"


def test_0013_denoiser_train_step(ns):
    import torch
    denoiser_train_step = ns["denoiser_train_step"]
    diffusion_loss = ns["diffusion_loss"]
    init_denoiser_params = ns["init_denoiser_params"]
    linear_beta_schedule = ns["linear_beta_schedule"]
    compute_alphas = ns["compute_alphas"]
    compute_alpha_bars = ns["compute_alpha_bars"]

    T, D, B, time_dim, lr = 20, 2, 16, 8, 0.05
    betas = linear_beta_schedule(T)
    alpha_bars = compute_alpha_bars(compute_alphas(betas))
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)
    torch.manual_seed(0)
    x0 = torch.randn(B, D)

    # Snapshot params, then take a step and verify the SGD update by hand.
    g = torch.Generator().manual_seed(42)
    before = [(W.detach().clone(), b.detach().clone()) for (W, b) in params]

    # Recompute loss + grads independently on a fresh copy of params.
    ref_params = [(W.detach().clone().requires_grad_(True),
                   b.detach().clone().requires_grad_(True)) for (W, b) in params]
    g_ref = torch.Generator().manual_seed(42)
    ref_loss = diffusion_loss(ref_params, x0, alpha_bars, T, time_dim, generator=g_ref)
    ref_loss.backward()
    expected_after = []
    with torch.no_grad():
        for (W, b) in ref_params:
            expected_after.append((W - lr * W.grad, b - lr * b.grad))

    returned = denoiser_train_step(params, x0, alpha_bars, T, time_dim, lr, generator=g)

    # Returned loss is a python float matching the recomputed loss.
    assert isinstance(returned, float), f"expected float, got {type(returned)}"
    assert abs(returned - float(ref_loss)) < 1e-5, f"{returned} vs {float(ref_loss)}"

    # Params updated in place to the expected SGD result.
    for (W, b), (eW, eb), (bW, bb) in zip(params, expected_after, before):
        assert torch.allclose(W, eW, atol=1e-5), "W update mismatch"
        assert torch.allclose(b, eb, atol=1e-5), "b update mismatch"
        # Params actually changed (grads were nonzero somewhere).
    moved = any(not torch.allclose(W, bW, atol=1e-8) for (W, b), (bW, bb) in zip(params, before))
    assert moved, "no parameter moved after a train step"


def test_0014_train_denoiser(ns):
    import torch
    train_denoiser = ns["train_denoiser"]
    init_denoiser_params = ns["init_denoiser_params"]
    linear_beta_schedule = ns["linear_beta_schedule"]
    compute_alphas = ns["compute_alphas"]
    compute_alpha_bars = ns["compute_alpha_bars"]

    T, D, B, time_dim, lr = 20, 2, 16, 8, 0.05
    betas = linear_beta_schedule(T)
    alpha_bars = compute_alpha_bars(compute_alphas(betas))
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)
    torch.manual_seed(0)
    # Structured target so there is signal to learn.
    x0 = torch.randn(B, D) * 0.5 + torch.tensor([2.0, -1.0])

    n_steps = 200
    g = torch.Generator().manual_seed(0)
    history = train_denoiser(params, x0, alpha_bars, T, time_dim, lr, n_steps, generator=g)

    assert isinstance(history, list), "history must be a list"
    assert len(history) == n_steps, f"expected {n_steps} losses, got {len(history)}"
    assert all(isinstance(v, float) for v in history), "history entries must be floats"
    assert all(torch.isfinite(torch.tensor(v)) for v in history), "non-finite loss in history"

    # Loss must trend down on the fixed batch.
    assert history[-1] < history[0], f"loss did not decrease: {history[0]} -> {history[-1]}"
    early = sum(history[:20]) / 20.0
    late = sum(history[-20:]) / 20.0
    assert late < early, f"tail-mean loss not below head-mean: {early} -> {late}"

def test_0015_predict_x0_from_noise(ns):
    import torch
    predict_x0_from_noise = ns["predict_x0_from_noise"]
    q_sample = ns["q_sample"]
    torch.manual_seed(0)
    T, D, B = 20, 2, 16
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    x0 = torch.randn(B, D)
    noise = torch.randn(B, D)
    t = torch.randint(0, T, (B,))
    # round-trip: q_sample then invert must recover x0 exactly
    x_t = q_sample(x0, t, noise, alpha_bars)
    x0_hat = predict_x0_from_noise(x_t, t, noise, alpha_bars)
    assert x0_hat.shape == (B, D)
    assert torch.allclose(x0_hat, x0, atol=1e-5)
    # independent manual oracle for a single sample
    ab = alpha_bars[int(t[0])]
    expected = (x_t[0] - torch.sqrt(1 - ab) * noise[0]) / torch.sqrt(ab)
    assert torch.allclose(x0_hat[0], expected, atol=1e-5)


def test_0016_posterior_mean(ns):
    import torch
    posterior_mean = ns["posterior_mean"]
    torch.manual_seed(0)
    T, D, B = 20, 2, 16
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    x0 = torch.randn(B, D)
    x_t = torch.randn(B, D)
    # use a mix of t==0 and t>0
    t = torch.arange(B) % T
    out = posterior_mean(x0, x_t, t, betas, alphas, alpha_bars)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()
    # independent per-row oracle
    expected = torch.empty(B, D)
    for i in range(B):
        ti = int(t[i])
        beta_t = betas[ti]
        alpha_t = alphas[ti]
        ab_t = alpha_bars[ti]
        ab_prev = torch.tensor(1.0) if ti == 0 else alpha_bars[ti - 1]
        coef0 = beta_t * torch.sqrt(ab_prev) / (1 - ab_t)
        coeft = (1 - ab_prev) * torch.sqrt(alpha_t) / (1 - ab_t)
        expected[i] = coef0 * x0[i] + coeft * x_t[i]
    assert torch.allclose(out, expected, atol=1e-5)


def test_0017_ddpm_sample_step(ns):
    import torch
    ddpm_sample_step = ns["ddpm_sample_step"]
    init_denoiser_params = ns["init_denoiser_params"]
    predict_noise = ns["predict_noise"]
    predict_x0_from_noise = ns["predict_x0_from_noise"]
    posterior_mean = ns["posterior_mean"]
    T, D, B, time_dim = 20, 2, 16, 8
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)
    torch.manual_seed(0)
    x_t = torch.randn(B, D)
    # at t==0 the step is deterministic (no noise added): equals posterior_mean of predicted x0
    out0 = ddpm_sample_step(params, x_t, 0, betas, alphas, alpha_bars, time_dim)
    t_batch = torch.zeros(B, dtype=torch.long)
    eps = predict_noise(params, x_t, t_batch, time_dim)
    x0_hat = predict_x0_from_noise(x_t, t_batch, eps, alpha_bars)
    expected0 = posterior_mean(x0_hat, x_t, t_batch, betas, alphas, alpha_bars)
    assert out0.shape == (B, D)
    assert torch.allclose(out0, expected0, atol=1e-5)
    assert torch.isfinite(out0).all()
    # at t>0 noise is added, so output differs from the deterministic mean
    g = torch.Generator().manual_seed(123)
    out5 = ddpm_sample_step(params, x_t, 5, betas, alphas, alpha_bars, time_dim, generator=g)
    assert out5.shape == (B, D)
    assert torch.isfinite(out5).all()
    tb = torch.full((B,), 5, dtype=torch.long)
    eps5 = predict_noise(params, x_t, tb, time_dim)
    x0h5 = predict_x0_from_noise(x_t, tb, eps5, alpha_bars)
    mean5 = posterior_mean(x0h5, x_t, tb, betas, alphas, alpha_bars)
    assert not torch.allclose(out5, mean5, atol=1e-5)


def test_0018_ddpm_sample_loop(ns):
    import torch
    ddpm_sample_loop = ns["ddpm_sample_loop"]
    init_denoiser_params = ns["init_denoiser_params"]
    T, D, B, time_dim = 20, 2, 16, 8
    betas = torch.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)
    g1 = torch.Generator().manual_seed(0)
    out1 = ddpm_sample_loop(params, (B, D), betas, alphas, alpha_bars, time_dim, generator=g1)
    assert out1.shape == (B, D)
    assert torch.isfinite(out1).all()
    # reproducible with same generator seed
    g2 = torch.Generator().manual_seed(0)
    out2 = ddpm_sample_loop(params, (B, D), betas, alphas, alpha_bars, time_dim, generator=g2)
    assert torch.allclose(out1, out2, atol=1e-5)
    # different seed yields a different draw
    g3 = torch.Generator().manual_seed(1)
    out3 = ddpm_sample_loop(params, (B, D), betas, alphas, alpha_bars, time_dim, generator=g3)
    assert not torch.allclose(out1, out3, atol=1e-5)

def test_0019_ddim_sample_step(ns):
    import torch
    ddim_sample_step = ns["ddim_sample_step"]
    predict_noise = ns["predict_noise"]
    predict_x0_from_noise = ns["predict_x0_from_noise"]
    gather_at_timesteps = ns["gather_at_timesteps"]
    init_denoiser_params = ns["init_denoiser_params"]
    linear_beta_schedule = ns["linear_beta_schedule"]
    compute_alphas = ns["compute_alphas"]
    compute_alpha_bars = ns["compute_alpha_bars"]

    T, D, B, time_dim = 20, 2, 16, 8
    betas = linear_beta_schedule(T)
    alpha_bars = compute_alpha_bars(compute_alphas(betas))
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)

    torch.manual_seed(0)
    x_t = torch.randn(B, D)
    t, t_prev = 10, 7

    # Independent oracle replicating the DDIM eta=0 formula.
    t_batch = torch.full((B,), t, dtype=torch.long)
    with torch.no_grad():
        eps = predict_noise(params, x_t, t_batch, time_dim)
        x0_hat = predict_x0_from_noise(x_t, t_batch, eps, alpha_bars)
        ab_prev = gather_at_timesteps(alpha_bars, torch.full((B,), t_prev, dtype=torch.long))
        expected = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps

    with torch.no_grad():
        out = ddim_sample_step(params, x_t, t, t_prev, alpha_bars, time_dim)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, expected, atol=1e-5)

    # t_prev < 0 -> alpha_bar_prev = 1 -> result equals x0_hat exactly.
    with torch.no_grad():
        out_last = ddim_sample_step(params, x_t, t, -1, alpha_bars, time_dim)
    assert torch.allclose(out_last, x0_hat, atol=1e-5)

    # Determinism: same inputs -> same output.
    with torch.no_grad():
        out2 = ddim_sample_step(params, x_t, t, t_prev, alpha_bars, time_dim)
    assert torch.allclose(out, out2, atol=1e-6)


def test_0020_ddim_sample_loop(ns):
    import torch
    ddim_sample_loop = ns["ddim_sample_loop"]
    init_denoiser_params = ns["init_denoiser_params"]
    linear_beta_schedule = ns["linear_beta_schedule"]
    compute_alphas = ns["compute_alphas"]
    compute_alpha_bars = ns["compute_alpha_bars"]

    T, D, B, time_dim = 20, 2, 16, 8
    betas = linear_beta_schedule(T)
    alpha_bars = compute_alpha_bars(compute_alphas(betas))
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)

    with torch.no_grad():
        torch.manual_seed(123)
        a = ddim_sample_loop(params, (B, D), alpha_bars, time_dim, n_steps=5)
        torch.manual_seed(123)
        b = ddim_sample_loop(params, (B, D), alpha_bars, time_dim, n_steps=5)

    assert a.shape == (B, D)
    assert torch.isfinite(a).all()
    # DDIM is deterministic given the same starting noise (same seed).
    assert torch.allclose(a, b, atol=1e-6)


def test_0021_generate_samples(ns):
    import torch
    generate_samples = ns["generate_samples"]
    init_denoiser_params = ns["init_denoiser_params"]
    linear_beta_schedule = ns["linear_beta_schedule"]
    compute_alphas = ns["compute_alphas"]
    compute_alpha_bars = ns["compute_alpha_bars"]

    T, D, time_dim = 20, 2, 8
    betas = linear_beta_schedule(T)
    alphas = compute_alphas(betas)
    alpha_bars = compute_alpha_bars(alphas)
    params = init_denoiser_params(D, time_dim, [64, 64], seed=0)
    n = 16

    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        s_ddpm = generate_samples(params, n, D, betas, alphas, alpha_bars, time_dim, method="ddpm", generator=g)
        s_ddim = generate_samples(params, n, D, betas, alphas, alpha_bars, time_dim, method="ddim", ddim_steps=5)

    assert s_ddpm.shape == (n, D)
    assert s_ddim.shape == (n, D)
    assert torch.isfinite(s_ddpm).all()
    assert torch.isfinite(s_ddim).all()

    # DDIM determinism via seeded global RNG (no generator arg used by ddim loop).
    with torch.no_grad():
        torch.manual_seed(7)
        d1 = generate_samples(params, n, D, betas, alphas, alpha_bars, time_dim, method="ddim", ddim_steps=4)
        torch.manual_seed(7)
        d2 = generate_samples(params, n, D, betas, alphas, alpha_bars, time_dim, method="ddim", ddim_steps=4)
    assert torch.allclose(d1, d2, atol=1e-6)
