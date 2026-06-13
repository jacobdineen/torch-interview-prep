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
    "35": (
        "You exercised the `Dataset` contract: `__len__` + `__getitem__`, then a seeded "
        "`DataLoader` over it. Real-world use: EVERY training loop sits on this pair; "
        "shuffling reproducibly means seeding a `torch.Generator`, not `random`. Gotcha: "
        "`__getitem__` should return tensors of consistent shape/dtype, or the default "
        "collate will throw on the first ragged batch."
    ),
    "36": (
        "You wrote a collate_fn that pads ragged sequences into a (B, T_max) batch with a "
        "bool mask. Real-world use: any text/audio batch needs this; it's what "
        "`pad_sequence` + an attention mask do in every NLP pipeline. Gotcha: the mask "
        "must mark REAL tokens (not padding), and downstream means/losses must use it — "
        "padding that leaks into a mean is a silent metric bug."
    ),
    "37": (
        "You assembled an MLP with `nn.Sequential` and ran a fit loop: forward, loss, "
        "`backward()`, `step()`, `zero_grad()`. Real-world use: this five-line cadence IS "
        "model training — everything fancier is wrapped around it. Gotcha: forgetting "
        "`zero_grad()` makes gradients accumulate across steps, which looks like a "
        "mysteriously unstable loss."
    ),
    "38": (
        "You implemented gradient accumulation: divide each micro-batch loss by the "
        "number of accumulation steps and only `step()` after the last one. Real-world "
        "use: it's how a 8GB GPU trains with an effective batch of 256. Gotcha: forgetting "
        "the 1/N scaling gives gradients N× too large — same symptom as a too-high LR."
    ),
    "39": (
        "You used `torch.autocast` (+ `GradScaler`) for mixed precision: matmuls run in "
        "fp16/bf16, reductions stay fp32, and the scaler keeps tiny fp16 grads from "
        "flushing to zero. Real-world use: AMP is on by default in serious training — "
        "2x throughput, half the memory. Gotcha: keep MASTER weights in fp32; with bf16 "
        "a scaler is unnecessary, with fp16 it's mandatory."
    ),
    "40": (
        "You captured intermediate activations with forward hooks "
        "(`module.register_forward_hook`). Real-world use: feature extraction, probing, "
        "Grad-CAM, activation stats — all without touching the model's code. Gotcha: "
        "hooks return handles; not calling `handle.remove()` leaks memory and makes "
        "every later forward slower."
    ),
    "43": (
        "You reproduced data-parallel semantics: each replica computes grads on its shard, "
        "then grads are AVERAGED across replicas so the update matches the full-batch one. "
        "Real-world use: this is exactly what DDP's all-reduce does after backward. "
        "Gotcha: averaging (not summing) is what keeps the LR meaning the same as "
        "single-GPU training."
    ),
    "44": (
        "You implemented conv2d by hand: slide a kernel over (H, W), dot-product each "
        "window, sum over input channels. Real-world use: understanding stride/padding "
        "arithmetic is how you debug shape errors in any CNN. Gotcha: output size is "
        "floor((H + 2p - k)/s) + 1 — an off-by-one here is the classic conv bug."
    ),
    "45": (
        "You implemented max-pooling: same sliding-window arithmetic as conv, but taking "
        "a max instead of a dot product. Real-world use: downsampling + translation "
        "tolerance in classic CNNs; modern nets often swap it for strided conv. Gotcha: "
        "pooling has NO learnable parameters — padding pads with -inf semantics, not zero."
    ),
    "46": (
        "You did conv shape arithmetic: 3x3/s1/p1 preserves size, stride divides it, "
        "dilation inflates the effective kernel to k + (k-1)(d-1), and transposed conv "
        "inverts the mapping for upsampling. Real-world use: designing encoders/decoders "
        "(UNets, GANs) is exactly this arithmetic. Gotcha: transposed conv output is "
        "(H-1)s - 2p + k — and its checkerboard artifacts are why many decoders use "
        "resize-then-conv instead."
    ),
    "47": (
        "You assembled a small CNN classifier end to end: conv/ReLU/pool stacks, a "
        "flatten, a linear head producing (B, 10) logits, and `train()`/`eval()` modes. "
        "Real-world use: this is the LeNet/VGG template every vision model refines. "
        "Gotcha: compute the flattened feature size from the conv arithmetic — hardcoding "
        "it breaks the moment the input resolution changes."
    ),
    "48": (
        "You implemented an RNN cell (tanh(Wx + Uh + b)) and unrolled it over time. "
        "Real-world use: the unroll-and-carry-state pattern survives in every recurrent "
        "decoder and in scan-style state-space models. Gotcha: parameters must be named "
        "and shaped like `nn.RNNCell` (W_ih, W_hh, b_ih, b_hh) for weight transplants "
        "to line up."
    ),
    "49": (
        "You implemented an LSTM cell: four gates (i, f, g, o) from one fused matmul, "
        "then c' = f*c + i*g and h' = o*tanh(c'). Real-world use: LSTMs still power "
        "production speech/time-series stacks; the gate algebra is interview canon. "
        "Gotcha: PyTorch's gate ORDER in the fused weight is i, f, g, o — scrambling it "
        "matches shapes but produces garbage."
    ),
    "50": (
        "You padded ragged sequences and then computed mask-aware reductions: a masked "
        "mean and the last REAL timestep per sequence. Real-world use: sentence "
        "embeddings, pooled classifier heads — anything that summarizes variable-length "
        "sequences. Gotcha: `output[:, -1]` is the last PADDED step, not the last real "
        "one; gather with lengths-1 instead."
    ),
    "51": (
        "You ran a bidirectional LSTM over packed sequences: `pack_padded_sequence` skips "
        "the padding compute, and the output concatenates forward+backward states into "
        "(B, T, 2H). Real-world use: the standard pre-transformer encoder (ELMo, BiLSTM-"
        "CRF). Gotcha: the 'sentence vector' is the concat of the forward LAST real state "
        "and the backward FIRST — not output[:, -1]."
    ),
    "52": (
        "You built sinusoidal positional encodings: PE[pos, 2i] = sin(pos/10000^(2i/d)), "
        "cos for odd dims, ADDED to the token embeddings. Real-world use: the original "
        "Transformer recipe, and the mental baseline for RoPE/ALiBi comparisons. Gotcha: "
        "the frequency exponent uses the PAIR index 2i/d — using the raw dim index halves "
        "every wavelength."
    ),
    "54": (
        "You built causal masking: a strictly-upper-triangular True mask, applied as "
        "-inf BEFORE softmax so position t attends only to ≤ t. Real-world use: this is "
        "what makes a decoder autoregressive; every GPT forward applies it. Gotcha: mask "
        "with -inf (or a large negative), never 0 — zeroing post-softmax breaks the "
        "probability normalization."
    ),
    "55": (
        "You assembled multi-head attention as a module: project to Q/K/V, reshape to "
        "(B, h, T, d_k), SDPA per head, concat, output-project. Real-world use: this "
        "module IS the transformer's workhorse; `nn.MultiheadAttention` is exactly this. "
        "Gotcha: the reshape is view(B, T, h, d_k).transpose(1, 2) — transposing before "
        "the view silently shuffles features across heads."
    ),
    "57": (
        "You implemented ALiBi: per-head slopes (a geometric sequence) times a "
        "-|i - j| distance bias added to attention scores — no learned positional "
        "parameters at all. Real-world use: trains at 1k context, extrapolates further; "
        "used by BLOOM/MPT. Gotcha: slopes are per-HEAD (shape (h, 1, 1) when adding) — "
        "broadcasting a single slope across heads removes the multi-scale effect."
    ),
    "59": (
        "You built a transformer ENCODER block: pre-LN residual attention, then a pre-LN "
        "residual MLP. Real-world use: BERT/ViT are stacks of exactly this block. "
        "Gotcha: pre-LN (norm inside the residual branch) is what makes deep stacks "
        "trainable without warmup tricks; post-LN looks similar and diverges at depth."
    ),
    "60": (
        "You implemented SwiGLU: down(silu(gate(x)) * up(x)) — a GATED MLP where one "
        "projection modulates the other elementwise. Real-world use: LLaMA-class models "
        "all swapped GELU MLPs for SwiGLU. Gotcha: there are THREE projections (gate, up, "
        "down), conventionally bias-free, with the hidden dim ~2/3 of the GELU "
        "equivalent to keep parameters matched."
    ),
    "61": (
        "You built a GPT DECODER block: causal self-attention + MLP, each in a pre-LN "
        "residual. Real-world use: GPT-2/LLaMA are N of these blocks in a trenchcoat. "
        "Gotcha: verify causality empirically — perturb a later token and assert earlier "
        "outputs are bit-identical; shape checks can't catch a leaked mask."
    ),
    "63": (
        "You built a sliding-window attention mask: position t sees only the last W "
        "tokens (optionally plus a global prefix). Real-world use: Mistral-style local "
        "attention — O(T·W) memory instead of O(T²). Gotcha: the window is applied ON TOP "
        "of causality; a window mask alone still leaks the future."
    ),
    "64": (
        "You tied the input embedding and the LM head: logits = h @ E.T, so one matrix "
        "serves both directions. Real-world use: GPT-2 and most LLMs tie weights — it "
        "saves V×d parameters and improves rare-token gradients. Gotcha: tie the actual "
        "TENSOR (same object), not a copy — `head.weight = emb.weight`, and check "
        "`head.weight is emb.weight` after loading checkpoints."
    ),
    "66": (
        "You implemented sampling filters: top-k keeps the k best logits, top-p keeps the "
        "smallest set whose probabilities sum past p — everything else goes to -inf "
        "before sampling. Real-world use: these two knobs are the body of every "
        "`generate()` call. Gotcha: top-p sorts DESCENDING and must keep the first token "
        "that crosses the threshold (shift the cutoff by one), or p=0.9 can return an "
        "empty set."
    ),
    "67": (
        "You applied a repetition penalty: divide positive logits (multiply negative "
        "ones) for tokens already generated, exactly once per unique token. Real-world "
        "use: the CTRL/HF `repetition_penalty` — the cheap fix for the model looping. "
        "Gotcha: penalizing a NEGATIVE logit means multiplying by the penalty (making it "
        "more negative) — dividing would make repeated tokens MORE likely."
    ),
    "68": (
        "You implemented beam search: keep the top-B partial hypotheses by cumulative "
        "log-prob, expand each by its top tokens, optionally length-normalize finished "
        "ones. Real-world use: translation/summarization decoders; the contrast with "
        "sampling is interview canon. Gotcha: compare hypotheses by SUM of log-probs "
        "(or a length-normalized score) — comparing raw products underflows, and "
        "unnormalized sums favor short outputs."
    ),
    "69": (
        "You wrote KV-cached greedy decoding: prefill the prompt once, then each step "
        "feeds ONE token and appends its K/V to the cache. Real-world use: this is why "
        "generation is O(T) per token instead of O(T²) — every serving stack does it. "
        "Gotcha: with a cache the causal mask for the new token is just 'attend to "
        "everything cached'; re-applying the full triangular mask breaks the step shapes."
    ),
    "70": (
        "You implemented speculative decoding: a small draft model proposes K tokens, the "
        "target model verifies them in ONE forward, accepting until the first "
        "disagreement (plus one corrected token). Real-world use: 2-3x serving speedups "
        "with mathematically IDENTICAL output distribution. Gotcha: on rejection you must "
        "resample from the residual distribution p_target - p_draft (clamped), not just "
        "take target's argmax."
    ),
    "71": (
        "You implemented InfoNCE: similarity matrix of two augmented views, temperature-"
        "scaled, cross-entropy against the diagonal (each sample's positive is its own "
        "other view). Real-world use: SimCLR/CLIP pretraining objectives. Gotcha: "
        "normalize embeddings first and divide by temperature τ — unnormalized dot "
        "products let one large-norm sample dominate the batch."
    ),
    "72": (
        "You implemented knowledge distillation: alpha-blend hard-label CE with "
        "KL(student_T || teacher_T) at temperature T, scaling the KL term by T². "
        "Real-world use: DistilBERT-style compression; alpha=1 must reduce to plain CE. "
        "Gotcha: the T² factor compensates the 1/T² gradient shrink from soft targets — "
        "dropping it silently underweights the teacher."
    ),
    "73": (
        "You implemented the VAE reparameterization trick: z = mu + sigma * eps with "
        "eps ~ N(0,1), making the sample differentiable w.r.t. mu/sigma, plus the "
        "closed-form KL to the unit Gaussian. Real-world use: VAEs, latent diffusion's "
        "first stage. Gotcha: the encoder outputs LOG-variance for stability — "
        "sigma = exp(0.5 * logvar), and the KL is -0.5 * sum(1 + logvar - mu² - "
        "exp(logvar))."
    ),
    "74": (
        "You implemented Gumbel-softmax: add Gumbel(0,1) noise (-log(-log(U))) to "
        "logits, softmax at temperature tau, optionally straight-through to one-hot. "
        "Real-world use: differentiable sampling over discrete choices — NAS, VQ "
        "alternatives, discrete latents. Gotcha: clamp U away from 0/1 before the "
        "double log, and remember tau→0 sharpens toward argmax while killing gradients."
    ),
    "76": (
        "You used gradient checkpointing: drop intermediate activations in forward and "
        "recompute them during backward, trading ~30% compute for O(sqrt(L)) activation "
        "memory. Real-world use: how long-context transformer training fits in memory at "
        "all. Gotcha: the checkpointed function must be side-effect-free w.r.t. RNG "
        "(dropout needs `preserve_rng_state`, which `torch.utils.checkpoint` does by "
        "default) or recomputation diverges from the original forward."
    ),
}


def get_concept(num):
    """Return the concept blurb for a problem ID (e.g. '03' or '03b'), or None.
    Children inherit their parent's concept blurb."""
    import re as _re
    if num in CONCEPTS:
        return CONCEPTS[num]
    m = _re.match(r"^(\d+)", str(num))
    if m:
        return CONCEPTS.get(m.group(1))
    return None
