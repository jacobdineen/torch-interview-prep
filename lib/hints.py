"""Per-parent hint lists shown by `check.py NN --hint`.

Hints are graduated: each `--hint` invocation reveals the next one. Child problems
(e.g. p04b) inherit their parent's (p04) hints. Hint counter is persisted in
`.hint_state.json` (gitignored, per-machine). Reset with `--reset-hints`.
"""

HINTS = {
    # ====================================================================
    # TIER 1 — Tensor Fundamentals
    # ====================================================================
    "01": [
        "All four are one-liners using built-in torch creation helpers.",
        "`torch.zeros(shape, dtype=...)` takes shape as a tuple/list, NOT as a tensor.",
        "Use `torch.arange(start, end, step)`. To move/cast: `x.to(dtype=..., device=...)`.",
        "tensor_info should return tuple(x.shape) (not torch.Size), x.dtype, x.device, x.numel().",
    ],
    "02": [
        "All indexing is direct: `x[i]`, `x[:, ::2]`, `x[idx]`, `x[:k, :k]`.",
        "`torch.diag(x)` (or `x.diagonal()`) gets the diagonal of a square matrix.",
        "select_rows with a LongTensor uses fancy indexing: `x[idx]`.",
    ],
    "03": [
        "`x.reshape(B, -1)` for flatten; `x.transpose(-1, -2)` for swap; `x.permute(...)` for arbitrary reorder.",
        "`permute` arguments are the SOURCE indices in OUTPUT order. (B,C,H,W) -> (B,H,W,C) is permute(0, 2, 3, 1).",
        "`.contiguous()` returns the same tensor if already contiguous, or a contiguous copy otherwise.",
    ],
    "04": [
        "Always reach for `keepdim=True` when the reduction result will be used in broadcasting.",
        "For pairwise distances, expand: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b. Use A @ B.T for the dot products.",
        "For per-row normalization, use `x.mean(dim=1, keepdim=True)` and `x.std(dim=1, unbiased=False, keepdim=True)`.",
        "Outer product of u (M,) and v (N,) — use `torch.outer(u, v)` OR `u[:, None] * v[None, :]`.",
    ],
    "05": [
        "`dim=` accepts a tuple, e.g. `x.mean(dim=(0, 2, 3))` reduces over multiple dims at once.",
        "For `(B, *)` -> `(B,)` reductions, `x.flatten(1).max(dim=1).values` works for any rank.",
        "Tensor.max returns a named tuple `(values, indices)`. Use `.values` when you want just the numbers.",
        "`x.argmax(dim=-1)` returns the index of the max along the last dim — no `.values/.indices`.",
    ],
    "06": [
        "`torch.cat` joins along an EXISTING dim; `torch.stack` creates a NEW dim.",
        "`torch.split(x, k)` splits into pieces OF SIZE k. To split into k pieces, use `torch.chunk(x, k)`.",
        "Interleave: `torch.stack([a, b], dim=1).reshape(-1, D)` because reshape walks the rightmost dims first.",
    ],
    "07": [
        "`torch.where(condition, value_if_true, value_if_false)` — signature in that order. Easy to flip backwards.",
        "Counts of booleans: `(bool_tensor).sum().item()` gives the count of True values as a Python int.",
        "`torch.nan_to_num(x, nan=value)` replaces NaN with value. Or `torch.where(torch.isnan(x), value, x)`.",
        "`x[mask].mean()` selects elements where the bool mask is True, then averages.",
    ],
    "08": [
        "`gather` READS values by index along a dim. The index tensor's shape must match the input except on the gather dim.",
        "gather_per_row(x, idx): `x.gather(1, idx.unsqueeze(1)).squeeze(1)` — unsqueeze idx to (N, 1), then squeeze the result.",
        "one_hot is `zeros + scatter_(1, labels.unsqueeze(1), 1.0)`. scatter_ writes values at index positions.",
        "For segment-sum (sum rows by index), `out.index_add_(0, index, values)` is cleaner than scatter_add_.",
    ],
    "09": [
        "Every letter in the einsum string is one dim. Letters not in the OUTPUT after `->` get summed away.",
        "matmul: `'ik,kj->ij'` (k contracts). Letters are arbitrary; what matters is which repeat.",
        "Use `...` for any-leading-dims: `'...ij,...jk->...ik'` is a matmul that works for any batch shape.",
        "For attention scores: q has T_q, k has T_k — they need different letters (e.g. i, j). d contracts.",
    ],
    "10": [
        "`x.topk(k)` returns (values, indices) named tuple; defaults are dim=-1, largest=True, sorted=True.",
        "For k SMALLEST values' indices, pass `largest=False` to topk and take `.indices`.",
        "For 2D global argmax: `x.argmax()` returns a flat index; recover (row, col) with `// x.shape[1]` and `% x.shape[1]`.",
        "k-th largest: `x.flatten().topk(k).values[-1]` is O(N log k), faster than a full sort.",
    ],

    # ====================================================================
    # TIER 2 — Autograd & Linear
    # ====================================================================
    "11": [
        "`tensor.backward()` accumulates into `.grad` on leaf tensors. Inputs need requires_grad=True.",
        "`torch.autograd.grad(y, x, grad_outputs=...)` returns gradients directly without populating .grad.",
        "For higher-order derivatives, pass `create_graph=True` to the first grad call so the graph is retained.",
        "Diagonal of the Jacobian of an elementwise function f(z): `grad(f(z).sum(), z)` (since off-diagonals are 0).",
    ],
    "12": [
        "Custom torch.autograd.Function needs @staticmethod forward and @staticmethod backward.",
        "Use `ctx.save_for_backward(x)` in forward, then `(x,) = ctx.saved_tensors` in backward.",
        "Stable softplus forward: `max(x, 0) + log1p(exp(-|x|))`. Avoids overflow at large |x|.",
        "Stable softplus backward: dL/dx = grad_output * sigmoid(x). Verify with `torch.autograd.gradcheck`.",
    ],
    "13": [
        "`detach()` returns a new tensor sharing data but with no autograd history (requires_grad=False).",
        "`with torch.no_grad():` disables grad recording inside the block. Use for inference / parameter norm.",
        "`stop_gradient_at(x, t)`: use `torch.where(x > t, x.detach(), x)` so grads only flow where x <= t.",
        "`t.is_leaf` is True for tensors created by the user (not the result of an op). `t.requires_grad` is independent.",
    ],
    "14": [
        "nn.Linear stores weight as `(out_features, in_features)`. Forward is `x @ self.weight.t() + bias` for any-rank x.",
        "Default init in PyTorch nn.Linear: `nn.init.kaiming_uniform_(weight, a=math.sqrt(5))`.",
        "Bias default init: `nn.init.uniform_(bias, -1/sqrt(in_features), 1/sqrt(in_features))`.",
        "Handle bias=False by setting self.bias = None and skipping the addition in forward.",
    ],
    "15": [
        "ReLU: `torch.where(x > 0, x, torch.zeros_like(x))`. Leaky: replace zeros with `slope * x`.",
        "Stable sigmoid: branch on sign of x. For x>=0 use `1 / (1 + exp(-x))`; for x<0 use `exp(x) / (1 + exp(x))`.",
        "Stable tanh: `tanh(x) = sign(x) * (1 - exp(-2|x|)) / (1 + exp(-2|x|))`.",
        "Exact GELU: `x * 0.5 * (1 + erf(x / sqrt(2)))`. Use `torch.erf` and `math.sqrt(2)`.",
    ],
    "16": [
        "log-sum-exp trick: subtract max before exp to prevent overflow. `m = x.max(dim, keepdim=True).values`.",
        "softmax: `e = (x - m).exp(); return e / e.sum(dim, keepdim=True)`. Use keepdim so the divisor broadcasts.",
        "log_softmax: `(x - m) - logsumexp(x - m, dim, keepdim=True)`. Equivalent: `(x - m) - (x - m).exp().sum().log()`.",
        "All three use the same max-shift; only the post-processing differs. Verify against torch.softmax with allclose.",
    ],

    # ====================================================================
    # TIER 3 — Loss Functions
    # ====================================================================
    "17": [
        "MSE: `(pred - target) ** 2`, then mean/sum/none as the docstring asks.",
        "Huber: `|r| <= delta` -> `0.5 * r ** 2`; else `delta * (|r| - 0.5 * delta)`. Use `torch.where`.",
        "Compare to `F.mse_loss(p, t, reduction=red)` and `F.huber_loss(p, t, delta=..., reduction=red)` with allclose.",
    ],
    "18": [
        "Don't compute sigmoid then BCE separately — it's unstable. Fuse them.",
        "Stable form: `log(1 + exp(-x))` is `max_val + log(exp(-max_val) + exp(-x - max_val))` where max_val = max(-x, 0).",
        "With pos_weight: scale the positive log-sigmoid term. log_sigmoid(x) = -log(1 + exp(-x)).",
        "Match F.binary_cross_entropy_with_logits exactly. Test with extreme |x| values (1e4) — should be finite.",
    ],
    "19": [
        "Compute stable log_softmax: `(x - max(x)) - logsumexp(x - max(x), dim, keepdim=True)`.",
        "NLL: gather log-probs at target indices: `-log_probs.gather(1, target.unsqueeze(1)).squeeze(1)`.",
        "For ignore_index: build a bool mask `targets != ignore_index`, zero out invalid targets before gather, mask the NLL after.",
        "Mean reduction with ignore_index uses the count of NON-ignored examples as the divisor (not the full batch).",
    ],
    "20": [
        "smoothing=0 reduces to plain CE. smoothing=1 ignores targets entirely.",
        "Compute log_softmax once. nll = `-log_probs.gather(1, t.unsqueeze(1)).squeeze(1)`.",
        "smooth = `-log_probs.mean(dim=-1)`. Final loss = `(1-s) * nll + s * smooth`.",
        "Verify against `F.cross_entropy(logits, targets, label_smoothing=0.1)`.",
    ],
    "21": [
        "gamma=0 reduces to plain CE. Compute log_softmax once, then gather log_pt at target indices.",
        "pt = log_pt.exp(). focal = `-((1 - pt) ** gamma) * log_pt`.",
        "alpha can be a per-class tensor (index it with targets) OR a scalar applied uniformly OR None.",
    ],
    "22": [
        "Triplet: `(|a-p| - |a-n| + margin).clamp_min(0)` per example.",
        "For hardest-triplet: pairwise distances via `torch.cdist(emb, emb)` -> (N, N) matrix.",
        "Build `same_class` mask: `labels.unsqueeze(0) == labels.unsqueeze(1)`. Exclude diagonal for positives.",
        "Hardest pos = max distance among same-class (excl self). Hardest neg = min distance among different-class.",
    ],

    # ====================================================================
    # TIER 4 — Normalization
    # ====================================================================
    "23": [
        "BatchNorm1d: parameters weight (γ) and bias (β); buffers running_mean and running_var.",
        "Train mode: use BATCH stats for normalization (mean/var with `unbiased=False`); update running stats with `unbiased=True` var.",
        "Eval mode: use running_mean and running_var directly (no update).",
        "register_buffer for running stats so they get saved with state_dict and moved with .to(device).",
    ],
    "24": [
        "LayerNorm normalizes over the LAST `len(normalized_shape)` dims per sample.",
        "For `normalized_shape=(D,)`, dims=(-1,). For `(H, W)`, dims=(-2, -1). Build dims tuple in __init__.",
        "Use mean and var with `keepdim=True` so the result broadcasts back against x.",
        "No running stats. Use `unbiased=False` for normalization.",
    ],
    "25": [
        "RMSNorm has NO mean subtraction — only divide by RMS.",
        "rms = `sqrt(mean(x^2) + eps)` over the last dims. Then `x / rms * weight`.",
        "Weight is initialized to ones, shape == normalized_shape. No bias.",
        "Used in LLaMA, T5. Slightly faster than LayerNorm because no centering.",
    ],
    "26": [
        "Reshape x from `(N, C, *)` to `(N, G, C//G, *)` — chunk channels into groups.",
        "Normalize over (channel_within_group, *spatial) per (batch, group). Use `unbiased=False`.",
        "Reshape back to `(N, C, *)`, then apply per-channel affine (broadcast weight/bias of shape `(C,)`).",
        "With num_groups=1 it's InstanceNorm. With num_groups=C it's LayerNorm-style.",
    ],
    "27": [
        "Inverted dropout: at TRAIN time multiply by mask AND divide by (1 - p) so expected value is preserved.",
        "Eval mode is identity (no mask, no scaling).",
        "p=0 must be identity even in train mode. Use `torch.rand_like(x) >= self.p` for the mask.",
    ],

    # ====================================================================
    # TIER 5 — Embeddings & Init
    # ====================================================================
    "28": [
        "Embedding is just `self.weight[idx]`. Weight shape `(num_embeddings, embedding_dim)`.",
        "If padding_idx is set: zero that row at init AND register a backward hook to zero its gradient.",
        "Hook: `self.weight.register_hook(lambda g: g.clone().index_fill_(0, torch.tensor([pidx]), 0.0))`.",
    ],
    "29": [
        "Both functions are in-place: modify tensor and return tensor.",
        "Xavier: `std = gain * sqrt(2 / (fan_in + fan_out))`. For 2D weight, fan_in=shape[1], fan_out=shape[0].",
        "Kaiming relu: `gain=sqrt(2)`. Kaiming linear: `gain=1`. fan depends on mode ('fan_in' or 'fan_out').",
        "Use `tensor.normal_(0, std)` inside a `torch.no_grad()` block.",
    ],

    # ====================================================================
    # TIER 6 — Optimizers
    # ====================================================================
    "30": [
        "Plain SGD: `p.data -= lr * p.grad`. Use `@torch.no_grad()` decorator on step().",
        "Add weight decay (L2) as `g = g + wd * p` BEFORE applying momentum.",
        "Momentum buffer: `buf = mu * buf + g` (PyTorch default, not Nesterov). Then `p -= lr * buf`.",
        "Keep a dict `{id(p): buf}` for per-parameter momentum buffers.",
    ],
    "31": [
        "Adam update per parameter: m = β1*m + (1-β1)*g; v = β2*v + (1-β2)*g². Then bias-correct.",
        "Bias correction: m_hat = m / (1 - β1^t), v_hat = v / (1 - β2^t) where t is step count.",
        "Weight decay in classic Adam: `g = g + wd * p` BEFORE the moment updates (L2 form).",
        "Param update: `p -= lr * m_hat / (sqrt(v_hat) + eps)`.",
    ],
    "32": [
        "AdamW differs from Adam only in WHERE weight decay is applied.",
        "AdamW: `p *= (1 - lr * wd)` BEFORE the Adam step. Grad is NOT augmented with L2 term.",
        "Without that ordering, you'll get tiny numerical drift from torch.optim.AdamW (try ~5e-5 atol).",
        "The Adam step itself: m, v updates + bias correction + `p -= lr * m_hat / (sqrt(v_hat) + eps)`.",
    ],
    "33": [
        "step < warmup_steps: linear ramp `base_lr * step / warmup`.",
        "step in [warmup, total]: cosine decay. progress = (step - warmup) / (total - warmup), clamp to [0, 1].",
        "Cosine formula: `min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))`.",
        "step > total: clamp to min_lr. Handle tensor inputs with `torch.where` over the same logic.",
    ],
    "34": [
        "Filter params to only those with non-None grad first.",
        "Compute total grad norm: `sqrt(sum((p.grad ** 2).sum() for p in params))`.",
        "If total > max_norm: scale = max_norm / (total + eps). Apply in-place: `p.grad.mul_(scale)`.",
        "Return the total norm BEFORE clipping (so the caller can log it).",
    ],

    # ====================================================================
    # TIER 7 — Training Infrastructure
    # ====================================================================
    "35": [
        "Dataset needs __len__ and __getitem__(i) returning (x, y).",
        "For determinism use `torch.Generator().manual_seed(seed)` and pass to `torch.randn(... , generator=g)`.",
        "make_loader wraps a Dataset in DataLoader with the given batch_size, shuffle.",
        "epoch_mean: iterate loader, accumulate total and count, return tensor(total / count).",
    ],
    "36": [
        "Find T = max(s.shape[0] for s in seqs); B = len(seqs).",
        "padded = full((B, T), pad_value, dtype=seqs[0].dtype). Then copy each seq into row i.",
        "mask = zeros((B, T), dtype=bool). Set mask[i, :len_i] = True for each sequence.",
        "labels = stack([b[1] for b in batch]).",
    ],
    "37": [
        "build_mlp: loop hidden_dims, interleaving Linear and ReLU; final layer has no ReLU.",
        "train_one_epoch: shuffle with torch.randperm(N), iterate batches, run forward + backward + step.",
        "fit: build Adam optimizer, run epochs of train_one_epoch, return the last epoch's loss.",
    ],
    "38": [
        "zero_grad ONCE at the start.",
        "For each micro-batch: compute loss, divide by accum_steps before backward, accumulate the un-scaled loss for the return value.",
        "step() ONCE at the end.",
        "This gives the same averaged gradient as concatenating all micro-batches into one.",
    ],
    "39": [
        "Use `torch.autocast(device_type=device_type, dtype=dtype)` context manager.",
        "On CPU with bfloat16: no GradScaler needed (only float16-on-GPU needs scaling).",
        "Pattern: forward inside autocast, loss computed inside, backward + step OUTSIDE.",
        "Master weights stay float32 even though forward computes in bf16.",
    ],
    "40": [
        "Use `model.named_modules()` to map name → module.",
        "register_forward_hook(callback) where callback receives (module, input, output).",
        "Save handles returned by register_forward_hook so you can remove them in a try/finally.",
        "Critical: remove ALL hooks in finally, even if forward raised. This prevents leaks across calls.",
    ],
    "41": [
        "save: build a dict with 'model', 'optimizer', 'epoch', 'best_metric' keys and torch.save it.",
        "load: torch.load with weights_only=False (we're storing more than tensors), then load_state_dict into both.",
        "Return (epoch, best_metric) from load. Optimizer state restores moment buffers exactly.",
    ],
    "42": [
        "ema_update: for each (p_e, p_m) param pair, `p_e.mul_(decay).add_(p_m.detach(), alpha=1-decay)`.",
        "Also sync buffers (running stats from BN, etc.) — copy_, not lerp.",
        "Use @torch.no_grad() on ema_update so it doesn't track itself in autograd.",
        "EMAWrapper deep-copies the model at construction; mark all EMA params requires_grad_(False).",
    ],
    "43": [
        "zip(*[list(m.parameters()) for m in models]) gives you tuples of matched params across replicas.",
        "For each tuple, collect .grad values. If all None, skip; if mixed None/non-None, raise ValueError.",
        "Mean: `torch.stack(grads, dim=0).mean(dim=0)`. Copy back into each replica's .grad in-place.",
        "Use @torch.no_grad() so the modifications don't enter the graph.",
    ],

    # ====================================================================
    # TIER 8 — Convolutions
    # ====================================================================
    "44": [
        "F.unfold extracts sliding-window patches: shape (B, Cin*kH*kW, L) where L is the number of output positions.",
        "Reshape the weight to (Cout, Cin*kH*kW), then matmul: (Cout, KIN) @ (B, KIN, L) -> (B, Cout, L).",
        "If bias is not None: add it after, broadcast (1, Cout, 1).",
        "Reshape to (B, Cout, Hout, Wout) where Hout = (H + 2*p - kH) // stride + 1 (same for Wout).",
    ],
    "45": [
        "If padding > 0, pad with -inf so padded positions never win the max.",
        "Use `x.unfold(2, k, s).unfold(3, k, s)` to extract spatial patches: shape (B, C, Hout, Wout, k, k).",
        "Reshape last two dims to (k*k) and take .max(dim=-1).values.",
        "If stride is None, default to kernel_size.",
    ],
    "46": [
        "Conv formula: `floor((h + 2*p - d*(k-1) - 1) / s) + 1`.",
        "For chain, just apply the formula iteratively, propagating the spatial size.",
        "Transposed conv: `(h - 1) * s - 2*p + d*(k-1) + output_padding + 1`.",
    ],
    "47": [
        "Two conv blocks: each is Conv2d -> ReLU -> MaxPool2d(2). Channels: 1 -> 16 -> 32.",
        "After two pools on 28x28 input: 28 -> 14 -> 7. Spatial size at head input: 32 * 7 * 7 = 1568.",
        "Head: Flatten -> Linear(1568, 64) -> ReLU -> Linear(64, num_classes).",
        "count_parameters: sum p.numel() for p in model.parameters() if p.requires_grad.",
    ],

    # ====================================================================
    # TIER 9 — Sequences
    # ====================================================================
    "48": [
        "Vanilla RNN cell: h_next = tanh(x @ W_ih.T + b_ih + h @ W_hh.T + b_hh).",
        "Weight shapes mirror nn.RNNCell: W_ih (H, I), W_hh (H, H), biases (H,).",
        "Init weights with kaiming_uniform_(a=math.sqrt(5)) to match PyTorch's default.",
        "run_rnn: loop over time, accumulate hidden states, stack at the end. Return (outputs, h_T).",
    ],
    "49": [
        "Compute gates: `x @ W_ih.T + b_ih + h_p @ W_hh.T + b_hh` -> shape (B, 4H).",
        "Split into i, f, g, o with .chunk(4, dim=-1) — IN THAT ORDER (PyTorch's convention).",
        "Activations: sigmoid on i, f, o; tanh on g.",
        "c = f * c_p + i * g; h = o * tanh(c). Return (h, c).",
    ],
    "50": [
        "lengths = `torch.tensor([s.shape[0] for s in seqs])`. T = max length.",
        "padded = zeros((B, T, D)); mask = zeros((B, T), dtype=bool). Copy each sequence into its row.",
        "masked_mean_pool: `(padded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)`.",
        "last_real_state: index = lengths - 1; use gather along time dim to pick the last valid token's state.",
    ],
    "51": [
        "Use pack_padded_sequence with enforce_sorted=False so you don't need to sort by length.",
        "Run the BiLSTM on the packed input; out is packed too, h_n is (2*num_layers, B, H).",
        "pad_packed_sequence to recover (B, T, 2H) padded outputs. Padding positions will be zero.",
        "Sentence embedding = concat of h_n[0] (last forward) and h_n[1] (last backward). Shape (B, 2H).",
    ],

    # ====================================================================
    # TIER 10 — Attention Primitives
    # ====================================================================
    "52": [
        "pos in [0, T). i in [0, d/2). div = 10000^(2i/d). angles[t, i] = t / div[i].",
        "PE[:, 0::2] = sin(angles); PE[:, 1::2] = cos(angles). Shape (T, d_model).",
        "add_positional_encoding broadcasts PE over the batch dim: x + PE.unsqueeze(0).",
        "At pos=0: even cols are sin(0)=0, odd cols are cos(0)=1. Test this exactly.",
    ],
    "53": [
        "Scaled scores: `q @ k.transpose(-1, -2) / sqrt(d)` where d = q.shape[-1].",
        "Mask: True positions get -inf BEFORE softmax (so they become exactly 0 after).",
        "Softmax over the last dim (key dim). Multiply with v: `attn @ v` -> output.",
        "Return both output AND the attention weights (for inspection / visualization).",
    ],
    "54": [
        "causal_mask(T): `torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)`. True ABOVE the diagonal.",
        "apply_causal_mask: scores.masked_fill(mask, float('-inf')).",
        "causal_attention does the full forward: scale, mask, softmax, attn @ v.",
        "Test the causal property: perturb future k/v, earlier outputs should be unchanged.",
    ],
    "55": [
        "Project: q = w_q(x).view(B, T, H, D).transpose(1, 2) -> (B, H, T, D). Same for k, v.",
        "Scaled scores: `q @ k.transpose(-1, -2) / sqrt(D)` where D is head_dim.",
        "Mask broadcasting: handle 2D (T, T), 3D (B, T, T), or already-4D shapes via .dim() check.",
        "After softmax + `attn @ v`: transpose(1, 2), contiguous, view back to (B, T, d_model), then w_o.",
    ],
    "56": [
        "theta_i = 1 / base^(2i/d) for i in [0, d/2). pos = arange(T). angles = pos[:, None] * theta[None, :].",
        "Return (cos(angles), sin(angles)), both shape (T, d/2).",
        "apply_rope: split x into pairs (2i, 2i+1). Rotate: new_2i = x_2i*cos - x_2i+1*sin; new_2i+1 = x_2i*sin + x_2i+1*cos.",
        "RoPE is a rotation -> norm-preserving. <RoPE(q,m), RoPE(k,n)> depends only on (m-n).",
    ],
    "57": [
        "slopes[h] = 2^(-8h/H) for h = 1..H. Tensor of shape (H,).",
        "bias[h, i, j] = -slopes[h] * |i - j| (symmetric) or causal variant. Output (H, T, T).",
        "Diagonal must be 0 (no penalty for attending to self).",
        "Used by replacing positional embedding entirely; bias is added to attention scores before softmax.",
    ],
    "58": [
        "Q has H heads of dim D; K, V have G groups of dim D where G | H.",
        "Project: w_q outputs (H*D), w_k and w_v output (G*D). View to add the head/group axis.",
        "Expand K, V to H heads via `.repeat_interleave(H // G, dim=1)` along the group axis.",
        "G=1 is multi-query attention (MQA); G=H is plain MHA. LLaMA-2/3 use G between these.",
    ],

    # ====================================================================
    # TIER 11 — Transformer Architecture
    # ====================================================================
    "59": [
        "Pre-LN: ln1(x) -> attn -> add to x (residual). Then ln2(h) -> ff -> add to h.",
        "FF: Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model).",
        "Use nn.MultiheadAttention(batch_first=True, dropout=...) for the attention sublayer.",
        "Wrap each residual with Dropout: `x + drop(a)` and `h + drop(ff(...))`.",
    ],
    "60": [
        "SwiGLU is the modern LLM MLP: gated by silu(swish).",
        "Three linears, all bias=False: w_gate (d_model -> d_ff), w_up (d_model -> d_ff), w_down (d_ff -> d_model).",
        "Forward: `w_down(silu(w_gate(x)) * w_up(x))`. The * is elementwise — silu(gate) gates the up projection.",
    ],
    "61": [
        "Same structure as encoder block but BUILDS its own causal mask internally.",
        "T = x.shape[1]; mask = torch.triu(torch.ones(T, T, dtype=bool), diagonal=1).",
        "Pre-LN: ln1 -> attn(with causal mask) -> residual; ln2 -> mlp -> residual.",
        "MLP: Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model).",
    ],
    "62": [
        "Compute q, k_new, v_new from x_new only.",
        "If cache is None: k_all = k_new, v_all = v_new, Tpast = 0. Else: cat along time dim.",
        "Causal mask for the new positions: for each query at position (Tpast + i), allow keys at positions <= Tpast + i.",
        "Build mask as a (T_new, T_total) bool tensor and apply via masked_fill on the scores.",
    ],
    "63": [
        "Mask is True where MASKED OUT. Unmasked iff `j <= i AND i - j < window_size`.",
        "Build with broadcasting: i = arange(T).view(-1, 1); j = arange(T).view(1, -1).",
        "mask = ~((j <= i) & (i - j < window_size)).",
        "Sliding window attention = scaled dot product attention with this mask. window >= T == regular causal.",
    ],
    "64": [
        "Just an nn.Embedding(V, D); no separate lm_head Linear.",
        "Forward: h = embed(ids); logits = h @ embed.weight.T.",
        "The weight is shared by reference (same tensor for input and output projection).",
        "Test: only ONE Parameter in the model with shape (V, D) — saving the full lm_head's V*D params.",
    ],

    # ====================================================================
    # TIER 12 — Efficient Attention
    # ====================================================================
    "65": [
        "Process queries in blocks of size block_size_q. For each q block, iterate over k blocks.",
        "Maintain running max m, running denom l, and accumulator o per query row.",
        "Online softmax: when a new k block raises the max, rescale o and l by exp(m_old - m_new).",
        "After all k blocks: output = o / l. Mathematically identical to plain softmax(QK^T)V.",
    ],

    # ====================================================================
    # TIER 13 — Generation
    # ====================================================================
    "66": [
        "top_k_filter: get top-k values, use the smallest of them as a threshold, set lower entries to -inf.",
        "top_p: sort descending, cumsum the softmax probabilities, keep entries where cumprob <= p — but always keep the first.",
        "scatter the filtered logits back to their original positions using the sort indices.",
        "sample_token: divide logits by temperature, apply filters, softmax, multinomial sample.",
    ],
    "67": [
        "Iterate batch row by row. For each previously-generated token, look up its logit.",
        "If v > 0: divide by penalty. If v <= 0: multiply by penalty. Either way the post-softmax probability shrinks.",
        "apply_repetition_penalty repeats the operation if a token appears multiple times.",
        "penalize_unique applies it once per distinct token. Use set(gen[b].tolist()).",
    ],
    "68": [
        "Maintain a list of (tokens, total_logprob) beams. Start with the prompt.",
        "Each step: expand every beam by every top-k token, sort all candidates by total log-prob, take the top-k.",
        "When a beam emits end_token: move to a 'finished' list, score it (optionally with length penalty).",
        "After max_steps or no live beams: return the best finished beam.",
    ],
    "69": [
        "First call: feed the full prompt, get logits for every position. argmax over the last position is your first new token.",
        "Subsequent calls: feed ONLY the new token (1-step), use the returned cache to extend efficiently.",
        "If eos_token is given: track which rows have finished, pad with EOS, stop when all rows finish.",
        "Loop max_new_tokens times. Return the full (prompt + generated) sequence.",
    ],
    "70": [
        "Use the draft model to greedily extend by K tokens.",
        "Run the target model ONCE on the extended sequence (parallel verification of K + 1 positions).",
        "Walk the K positions: accept if target's argmax matches the draft. Stop at the first mismatch.",
        "Append the accepted prefix + a 'bonus' token (target's argmax at the first non-matched position).",
    ],

    # ====================================================================
    # TIER 14 — Auxiliary Losses
    # ====================================================================
    "71": [
        "Normalize both a and b along the last dim (F.normalize).",
        "logits = a_normalized @ b_normalized.T / tau. labels = arange(N) (identity matching).",
        "Symmetric InfoNCE: 0.5 * (CE(logits, labels) + CE(logits.T, labels)).",
        "At tau=1 with uncorrelated batch, loss ≈ log(N). With identical batch, loss ≈ 0.",
    ],
    "72": [
        "Compute teacher and student log-softmax at temperature T: log_softmax(logits / T).",
        "KL term: sum over class of teacher_p * (teacher_logp - student_logp); average over batch.",
        "Scale KL by T^2 (Hinton's correction so gradient magnitudes are temperature-invariant).",
        "Final loss: alpha * F.cross_entropy(student, targets) + (1 - alpha) * T^2 * KL.",
    ],
    "73": [
        "reparameterize: eps = randn_like(mu); z = mu + exp(0.5 * logvar) * eps.",
        "Gradient flows through mu and logvar via this formulation (the trick is sampling eps separately).",
        "KL closed form per element: 0.5 * (mu^2 + exp(logvar) - 1 - logvar).",
        "reduction='batchmean' divides per-element sum by batch size (PyTorch KLDivLoss convention).",
    ],
    "74": [
        "Gumbel noise: g = -log(-log(U)) where U ~ Uniform(0, 1). Add tiny eps to avoid log(0).",
        "Soft sample: softmax((logits + g) / tau). As tau -> 0 it concentrates on argmax.",
        "Hard: argmax to one-hot, then straight-through: y_hard - y_soft.detach() + y_soft.",
        "The detach trick gives one-hot forward but soft gradient backward.",
    ],

    # ====================================================================
    # TIER 15 — LLM Specifics & Capstone
    # ====================================================================
    "75": [
        "Wrap base nn.Linear; freeze its params (requires_grad_(False)).",
        "Two new params: lora_A shape (r, in), lora_B shape (out, r). Init A with kaiming_uniform_, B with zeros.",
        "B=0 at init means the adapter is identity — forward equals base forward.",
        "Forward: base(x) + (alpha/r) * x @ lora_A.T @ lora_B.T. merge_lora bakes the delta into a new Linear.",
    ],
    "76": [
        "Wrap blocks in nn.ModuleList. In train mode + input requires_grad: use torch.utils.checkpoint.checkpoint.",
        "Use_reentrant=False (newer PyTorch API). In eval mode, just call b(x) directly.",
        "Output and gradients must be numerically identical to a plain Sequential.",
        "Internally, checkpoint re-runs forward during backward instead of storing all activations.",
    ],
    "77": [
        "tok_embed (V, D); pos_embed (MAXT, D). Add embed(ids) + embed(arange(T)).",
        "Stack num_layers GPT decoder blocks. Each has its own LN + causal MHA + MLP.",
        "Final ln_f. Output logits = h @ tok_embed.weight.T — tied with input embed.",
        "causal_lm_loss: shift targets by 1; F.cross_entropy(logits[:, :-1], targets[:, 1:]).",
    ],
}


def get_hints(problem_id):
    """Return the list of hints for this problem id (parent-keyed lookup)."""
    import re as _re
    m = _re.match(r"^(\d+)", str(problem_id))
    if not m:
        return []
    return HINTS.get(m.group(1), [])
