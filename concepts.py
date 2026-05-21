"""Concept blurbs printed on PASS. Each entry is a short paragraph that:
  (1) names what you just exercised
  (2) points at where it shows up in real code
  (3) flags a common gotcha or forward link

Add or edit entries freely — the runner just looks them up by problem number.
"""

CONCEPTS = {
    "01": (
        "You exercised tensor creation + dtype/device handling. Real-world use: "
        "every model starts with `torch.zeros` / `torch.empty` for buffers, and "
        "every transfer to GPU is `.to(device)`. Gotcha: `torch.tensor(shape)` "
        "treats shape as DATA, not as a shape — use `torch.zeros(shape)` instead."
    ),
    "02": (
        "Indexing & slicing — the bread and butter of every batch/feature manipulation. "
        "Advanced indexing (`x[idx]` with a LongTensor) is how you implement embedding "
        "lookups and gather-style ops without explicit loops. Forward link: Problem 08 "
        "generalizes this with gather/scatter."
    ),
    "03": (
        "reshape/view/permute. `view` requires contiguous memory; `reshape` does it for "
        "you but may copy. `permute` reorders dims (its args are the SOURCE indices in "
        "output order). Gotcha: after `permute` the tensor is non-contiguous — call "
        "`.contiguous()` before `.view()` or use `.reshape()`."
    ),
    "04": (
        "Broadcasting + pairwise distance is the building block of attention "
        "(score = ||q - k||^2 in some formulations) and contrastive losses. Always "
        "prefer broadcasting over explicit batching — it's faster and clearer."
    ),
    "05": (
        "Reductions with `dim` and `keepdim` are everywhere — `mean(dim=1)` for "
        "sequence pooling, `sum(dim=-1)` after softmax for sanity checks. `keepdim=True` "
        "is crucial when you want to broadcast the result back: layernorm, batchnorm, "
        "softmax all use this pattern."
    ),
    "06": (
        "`cat` joins along an existing dim; `stack` creates a NEW dim. Confusing them "
        "is the classic shape-mismatch bug. `chunk(x, k)` is the inverse of `cat`; "
        "useful for splitting QKV after a single projection."
    ),
    "07": (
        "Boolean masking and `torch.where` are how you express conditional logic without "
        "Python branching (which would break vectorization and gradients). Used in "
        "every padding-aware attention mask, ReLU-from-scratch, and clip-by-value op."
    ),
    "08": (
        "`gather` picks values by index along a dim; `scatter` writes values to indices. "
        "Used in: cross-entropy NLL (`gather` log-probs at target indices), one-hot "
        "encoding (`scatter` ones into a zero tensor), segment sums (`scatter_add`). "
        "Modern alternative: `torch.index_select` for whole-row picks."
    ),
    "09": (
        "`einsum` is the most expressive tensor op — it can replace matmul, batched "
        "matmul, outer product, attention scores, and bilinear forms with one call. "
        "Index letters get summed if they don't appear in the output. Real use: "
        "attention is just `einsum('bhid,bhjd->bhij', q, k) / sqrt(d)`."
    ),
    "10": (
        "`topk`, `sort`, `argmax` — bedrock of sampling, beam search, and accuracy "
        "calculation (top-5 accuracy = `target in topk(logits, 5)`)."
    ),
    "11": (
        "Autograd basics. `tensor.backward()` populates `.grad` on leaf tensors. "
        "`torch.autograd.grad(...)` is the functional form (returns the gradient "
        "directly, doesn't accumulate in `.grad`). Use `create_graph=True` when you "
        "need higher-order derivatives (e.g., for MAML or Hessian estimation)."
    ),
    "12": (
        "Custom autograd.Function gives you a manual backward pass. Real uses: "
        "implementing flash attention CUDA kernels, straight-through estimators for "
        "discrete sampling, and any time the autodiff is slow or numerically bad. "
        "Always validate with `torch.autograd.gradcheck` on a double-precision input."
    ),
    "13": (
        "`detach()` gives you a fresh tensor that shares data but has no autograd "
        "history. `no_grad()` is a context manager that disables grad recording "
        "(faster forward, no gradient buildup). Use `no_grad` for inference / "
        "metric calculation; use `detach` for stop-gradient tricks in self-supervised "
        "learning (BYOL, SimSiam)."
    ),
    "14": (
        "nn.Linear is literally `x @ W.T + b`. Knowing this off the top of your head "
        "is interview table stakes. The weight shape `(out, in)` and the transpose in "
        "forward are PyTorch convention — JAX/Flax stores it as `(in, out)` instead."
    ),
    "15": (
        "Activations. ReLU is fast but dies; GELU is standard in transformers (smoother "
        "near zero); SiLU/Swish is in LLaMA. Sigmoid/Tanh are mostly historical except "
        "in LSTM gates. Numerical stability matters at the extremes — implement sigmoid "
        "as `where(x >= 0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))` to avoid overflow."
    ),
    "16": (
        "log-sum-exp trick: `log(sum(exp(x_i))) = m + log(sum(exp(x_i - m)))` where "
        "m = max(x). This is the *single most important numerical trick* in ML — every "
        "softmax, cross-entropy, and attention layer uses it under the hood."
    ),
    "17": (
        "MSE is L2 loss; Huber is L2 inside a tube and L1 outside it (robust to "
        "outliers). Used in: regression, RL value functions (Huber damps Q-value "
        "explosions). Smooth-L1 is Huber with delta=1."
    ),
    "18": (
        "Binary cross-entropy with logits. Always use `_with_logits` form — it fuses "
        "the sigmoid into the loss with the log-sum-exp trick, so it stays stable. "
        "`BCELoss(sigmoid(x), y)` is the unstable mistake nobody should make."
    ),
    "19": (
        "Multi-class cross-entropy = log-softmax + NLL. PyTorch's `F.cross_entropy` "
        "fuses these. `ignore_index` is how you mask out padding tokens in language "
        "models — set targets at PAD positions to -100 and they're excluded from "
        "the loss."
    ),
    "20": (
        "Label smoothing turns one-hot targets into `(1-eps)*one_hot + eps/C`. Helps "
        "calibration and acts as a soft regularizer. Used heavily in ImageNet training "
        "and in transformer pretraining. With smoothing=0 you get plain CE; with "
        "smoothing=1 the loss ignores the target entirely."
    ),
    "21": (
        "Focal loss: `-(1-p_t)^gamma * log(p_t)`. Down-weights well-classified examples. "
        "Designed for extreme class imbalance (RetinaNet, object detection). With "
        "gamma=0 it reduces to plain CE."
    ),
    "22": (
        "Triplet loss enforces `d(a, p) + margin < d(a, n)`. Used in face recognition, "
        "metric learning, and any embedding task where you have similar/dissimilar "
        "pairs but no class labels. Hardest-triplet mining picks the toughest examples "
        "in each batch — necessary because most random triplets are too easy."
    ),
    "23": (
        "BatchNorm normalizes across the batch+spatial dims, per channel. Train mode "
        "uses current batch stats; eval mode uses running averages. The classic bug: "
        "forgetting `.eval()` at inference, so single-image inference uses statistics "
        "from a batch of one (zero variance → garbage)."
    ),
    "24": (
        "LayerNorm normalizes within each sample over the last D dims. Used everywhere "
        "in transformers because it doesn't depend on batch size (unlike BN). The eps "
        "matters: too small and you get NaNs; too large and you damp the normalization."
    ),
    "25": (
        "RMSNorm drops the mean subtraction — just divides by RMS. Used in LLaMA, T5. "
        "Slightly faster than LayerNorm and works about as well. The trend in modern "
        "LLMs has been toward simpler norms (LayerNorm → RMSNorm)."
    ),
    "26": (
        "GroupNorm sits between LayerNorm (group=1) and InstanceNorm (group=C). "
        "Useful when batch size is small (BN unstable) and channels group meaningfully "
        "(e.g., 32 groups in ResNet-style nets)."
    ),
    "27": (
        "Inverted dropout — scale survivors by 1/(1-p) at train so the expected output "
        "matches eval. The 'old' form scaled at eval, which made it harder to switch "
        "modes. Always use the inverted form (which is what PyTorch does)."
    ),
    "28": (
        "Embedding lookup is just `weight[idx]`. The trick is `padding_idx`: its row "
        "stays at zero AND its gradient is zeroed so the optimizer never moves it. "
        "Critical for variable-length sequence batching."
    ),
    "29": (
        "Xavier (Glorot) keeps activation variance constant assuming linear/tanh; "
        "Kaiming (He) accounts for ReLU's half-rectification. Wrong init → vanishing "
        "or exploding gradients in deep networks. PyTorch's default for nn.Linear is "
        "kaiming_uniform with `a=sqrt(5)` which is roughly LeCun-style."
    ),
    "30": (
        "SGD with momentum: `v = mu*v + g; theta -= lr*v`. Momentum is exponential "
        "averaging of past gradients, which dampens oscillation in narrow ravines. "
        "Modern transformers use AdamW, but plain SGD+momentum is still common for "
        "ConvNets (ResNet, EfficientNet)."
    ),
    "31": (
        "Adam = momentum on both first and second moments, with bias correction. "
        "Adaptive per-parameter learning rate makes it the default for most "
        "experimentation. Weight decay in PyTorch's Adam is L2 — for proper decoupled "
        "decay use AdamW (Problem 32)."
    ),
    "32": (
        "AdamW separates weight decay from the gradient computation — decay multiplies "
        "the weight directly. This is what you want for transformer training: in Adam, "
        "the L2 term gets normalized by the second-moment scaling, which makes decay "
        "effectively zero for parameters with large gradients."
    ),
    "33": (
        "Cosine + linear warmup. Warmup: model is fragile at init, so ramp lr up "
        "gradually. Cosine: ends with small lr for fine adjustments. Used in basically "
        "every transformer pretraining run."
    ),
    "34": (
        "Gradient clipping by global norm: scales ALL gradients down if their combined "
        "norm exceeds max_norm. Crucial for RNNs and transformers where gradient spikes "
        "can wreck training. Typical value: max_norm=1.0."
    ),
    "41": (
        "Scaled dot-product attention. The `/sqrt(d)` keeps dot-product variance "
        "constant as head_dim grows — without it, softmax saturates on a single key "
        "and gradients vanish. This single function is the heart of every transformer."
    ),
    "42": (
        "Multi-head attention = run attention in parallel on H subspaces of dim D/H. "
        "Allows the model to attend to different things simultaneously (positional, "
        "syntactic, semantic). The output projection (w_o) mixes heads back together."
    ),
    "53": (
        "Same content as problem 41 (scaled dot-product attention) — it's the canonical "
        "form. The mask: True = masked-out position; the softmax then assigns it zero "
        "weight. This is how causal masking and padding masking both work."
    ),
    "56": (
        "RoPE rotates pairs of dimensions by a position-dependent angle. The key "
        "property: <RoPE(q, m), RoPE(k, n)> depends only on (m - n), not on absolute "
        "position. Means the model can generalize to longer sequences than it was "
        "trained on (with caveats — see ALiBi for an alternative)."
    ),
    "58": (
        "Grouped-query attention: H query heads share G key/value heads (G < H). "
        "Halves or quarters KV cache memory at inference. MQA (G=1) is the extreme "
        "case used in older models; LLaMA-2/3 use G=8."
    ),
    "62": (
        "KV cache: during autoregressive generation, K and V for past tokens never "
        "change, so cache them. Only compute Q, K, V for the NEW token, concat into "
        "the cache, then attend. Turns generation from O(T^2) per step to O(T)."
    ),
    "65": (
        "FlashAttention's secret: online softmax. Process keys in blocks, maintain "
        "a running max + running denominator, and incrementally update the output. "
        "The mathematical equivalence to plain softmax(QK^T)V is exact — FlashAttn "
        "is the same algebra, reorganized for cache locality on a GPU."
    ),
    "75": (
        "LoRA: instead of fine-tuning a (d_out, d_in) weight matrix, train a "
        "low-rank delta `B @ A` where B is (d_out, r) and A is (r, d_in). With r=8 "
        "you have ~r * (in + out) trainable params instead of in * out. The base is "
        "frozen; the adapter is merged at inference (no overhead)."
    ),
    "77": (
        "You built a working language model from scratch. The pieces: token embed + "
        "position embed → stack of decoder blocks → final LayerNorm → tied LM head. "
        "Training: cross-entropy on shifted targets (predict token t+1 given tokens "
        "0..t). This is GPT-2 in ~50 lines. Real LLaMA-class models differ in: "
        "RMSNorm instead of LayerNorm, SwiGLU MLP, RoPE positional, GQA, longer context."
    ),
}


def get_concept(num):
    """Return the concept blurb for a problem number string (e.g. '03'), or None."""
    return CONCEPTS.get(num)
