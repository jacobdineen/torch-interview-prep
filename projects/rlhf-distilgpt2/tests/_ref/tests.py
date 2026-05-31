"""Hidden step tests for rlhf-distilgpt2.

Each ``test_<id>_<name>(ns)`` grades one step; ``ns`` holds the reference
implementations with the user's function swapped in. Pure-tensor steps are graded
by value; model-dependent steps use a cached, shared distilgpt2. Assertion
messages carry markers ("shape mismatch", "values differ") for the runner's
likely-cause heuristic.
"""
import numpy as np
import torch
from contextlib import contextmanager

_CACHE = {}


def _tok():
    if "tok" not in _CACHE:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("distilgpt2")
        t.pad_token = t.eos_token
        _CACHE["tok"] = t
    return _CACHE["tok"]


def _model():
    if "model" not in _CACHE:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained("distilgpt2")
        m.eval()
        _CACHE["model"] = m
    return _CACHE["model"]


class _TinyOut:
    def __init__(self, logits):
        self.logits = logits


class _TinyLM(torch.nn.Module):
    """A minimal causal-LM stand-in (embedding -> linear) with a HF-like output,
    used to test training steps fast without loading/mutating distilgpt2."""

    def __init__(self, vocab=20, d=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, vocab)
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        return _TinyOut(self.head(self.emb(input_ids)))


def _tiny_batch(vocab=20, b=4, t=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (b, t), generator=g)
    labels = ids.clone()
    labels[:, 0] = -100  # mask first (prompt-ish) position
    return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones(b, t, dtype=torch.long)}


class _TinyReward(torch.nn.Module):
    """Mean-pooled embedding -> scalar reward; for testing reward_train_step."""

    def __init__(self, vocab=20, d=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, d)
        self.head = torch.nn.Linear(d, 1)

    def forward(self, input_ids):
        return self.head(self.emb(input_ids).mean(dim=1)).squeeze(-1)


# --------------------------- harness ---------------------------

@contextmanager
def step(label):
    try:
        yield
    except AssertionError as e:
        raise AssertionError(f"step {label!r}: {e}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def expect_eq(got, want):
    if got != want:
        raise AssertionError(f"got {got!r} but expected {want!r}")


def expect_true(cond, msg="expected True"):
    if not cond:
        raise AssertionError(msg)


def expect_shape(got, shape):
    g = tuple(got.shape)
    if g != tuple(shape):
        raise AssertionError(f"shape mismatch: got {g} vs want {tuple(shape)}")


def expect_allclose(got, want, atol=1e-5, rtol=1e-4):
    g, w = _np(got).astype(float), _np(want).astype(float)
    if g.shape != w.shape:
        raise AssertionError(f"shape mismatch: got {g.shape} vs want {w.shape}")
    if np.allclose(g, w, atol=atol, rtol=rtol, equal_nan=True):
        return
    d = np.abs(g - w)
    idx = np.unravel_index(int(np.nanargmax(d)), d.shape) if d.size else ()
    raise AssertionError(f"values differ: max abs diff={np.nanmax(d):.3e} at {idx}; "
                         f"got={g[idx]:.6g} want={w[idx]:.6g}")


# --------------------- Part 1 — Model Setup and Decoding ---------------------

def test_0001_load_distilgpt2_tokenizer(ns):
    tok = ns["load_distilgpt2_tokenizer"]()
    with step("tokenizer round-trips text"):
        ids = tok("hello world").input_ids
        expect_true(len(ids) > 0, "should produce token ids")
        expect_true("hello" in tok.decode(ids), "should decode back")


def test_0002_load_distilgpt2_model(ns):
    model = ns["load_distilgpt2_model"]()
    with step("distilgpt2 config (6 layers, vocab 50257)"):
        expect_eq(model.config.vocab_size, 50257)
        expect_true(isinstance(model, torch.nn.Module), "should be an nn.Module")


def test_0003_set_pad_token_to_eos(ns):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("distilgpt2")
    ns["set_pad_token_to_eos"](t)
    with step("pad token becomes eos token"):
        expect_eq(t.pad_token, t.eos_token)


def test_0004_generate_and_decode(ns):
    out = ns["generate_and_decode"](_model(), _tok(), "The capital of France is", 5)
    with step("greedy continuation of the prompt"):
        expect_true(isinstance(out, str), "should return a string")
        expect_true(out.startswith("The capital of France is"), f"got {out!r}")


def test_0005_greedy_decode(ns):
    with step("argmax token id"):
        expect_eq(int(ns["greedy_decode"](torch.tensor([0.1, 5.0, 0.2]))), 1)
        expect_allclose(ns["greedy_decode"](torch.tensor([[0.1, 5.0], [9.0, 0.0]])), [1, 0])


def test_0006_sample_with_temperature(ns):
    logits = torch.tensor([0.0, 0.0, 100.0, 0.0])  # peaked on index 2
    g = torch.Generator().manual_seed(0)
    with step("sampling concentrates on the peak"):
        expect_eq(int(ns["sample_with_temperature"](logits, 1.0, g)), 2)


def test_0007_top_k_filter(ns):
    logits = torch.tensor([1.0, 3.0, 2.0, 0.0])
    out = ns["top_k_filter"](logits, 2)
    with step("keeps the top-2, others -inf"):
        expect_true(torch.isinf(out[0]) and torch.isinf(out[3]), "low logits should be -inf")
        expect_allclose(out[1], 3.0)
        expect_allclose(out[2], 2.0)


def test_0008_top_p_filter(ns):
    # softmax([ln.6, ln.3, ln.1]) = [0.6, 0.3, 0.1]; p=0.8 keeps the top two (cum 0.9).
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1]))
    out = ns["top_p_filter"](logits, 0.8)
    with step("nucleus keeps the smallest set reaching p"):
        expect_true(torch.isinf(out[2]), "tail token should be removed")
        expect_true(torch.isfinite(out[0]) and torch.isfinite(out[1]), "nucleus tokens kept")


# --------------------- Part 2 — SFT Data Pipeline ---------------------

def test_0009_build_synthetic_instruction_dataset(ns):
    data = ns["build_synthetic_instruction_dataset"](20)
    with step("n examples with instruction/response"):
        expect_eq(len(data), 20)
        expect_true("instruction" in data[0] and "response" in data[0], "missing keys")


def test_0010_format_example(ns):
    out = ns["format_example"]({"instruction": "Q?", "response": "A"})
    with step("normalizes to prompt/response"):
        expect_eq(out["prompt"], "Q?")
        expect_eq(out["response"], "A")


def test_0011_apply_template(ns):
    out = ns["apply_template"]({"prompt": "Q?", "response": "A"})
    with step("full = prompt_text + response"):
        expect_true(out["full_text"] == out["prompt_text"] + "A", "full should append response")
        expect_true("Q?" in out["prompt_text"], "prompt text should contain the prompt")


def test_0012_tokenize_example(ns):
    templ = ns["apply_template"]({"prompt": "Q?", "response": "A"})
    out = ns["tokenize_example"](templ, _tok())
    with step("input_ids + a prompt_len that's a prefix"):
        expect_true(len(out["input_ids"]) >= out["prompt_len"] > 0, "prompt_len must be a prefix")
        expect_eq(out["input_ids"][:out["prompt_len"]], _tok()(templ["prompt_text"]).input_ids)


def test_0013_build_labels(ns):
    ids = [5, 6, 7]
    labels = ns["build_labels"](ids)
    with step("labels copy the input ids (independent copy)"):
        expect_eq(list(labels), [5, 6, 7])
        labels[0] = 99
        expect_eq(ids[0], 5)  # original unchanged


def test_0014_mask_prompt_labels(ns):
    out = ns["mask_prompt_labels"]([1, 2, 3, 4], 2, -100)
    with step("first prompt_len labels become ignore_index"):
        expect_eq(list(out), [-100, -100, 3, 4])


def test_0015_pad_batch(ns):
    out = ns["pad_batch"]([[1, 2], [3, 4, 5]], 0)
    with step("right-pads to (batch, maxlen)"):
        expect_shape(out, (2, 3))
        expect_allclose(out, [[1, 2, 0], [3, 4, 5]])


def test_0016_make_attention_mask(ns):
    out = ns["make_attention_mask"](torch.tensor([[1, 2, 0], [3, 0, 0]]), 0)
    with step("1 for real tokens, 0 for pad"):
        expect_allclose(out, [[1, 1, 0], [1, 0, 0]])


def test_0017_collate_lm_batch(ns):
    exs = [{"input_ids": [1, 2, 3], "prompt_len": 1},
           {"input_ids": [4, 5], "prompt_len": 1}]
    batch = ns["collate_lm_batch"](exs, pad_id=0, ignore_index=-100)
    with step("input_ids/labels/attention_mask shaped (2,3); prompt+pad masked"):
        expect_shape(batch["input_ids"], (2, 3))
        expect_shape(batch["labels"], (2, 3))
        expect_allclose(batch["attention_mask"], [[1, 1, 1], [1, 1, 0]])
        expect_allclose(batch["labels"], [[-100, 2, 3], [-100, 5, -100]])  # prompt + pad ignored


def test_0018_iterate_minibatches(ns):
    batches = ns["iterate_minibatches"](list(range(10)), 4)
    with step("splits into batches of the given size"):
        expect_eq([len(b) for b in batches], [4, 4, 2])


def test_0019_train_val_split(ns):
    tr, va = ns["train_val_split"](list(range(10)), 0.2)
    with step("90/10-style split"):
        expect_eq(len(tr), 8)
        expect_eq(len(va), 2)


# --------------------- Part 3 — SFT Training Loop ---------------------

def test_0020_shift_logits_and_labels(ns):
    logits = torch.randn(2, 4, 5)
    labels = torch.arange(8).reshape(2, 4)
    sl, slab = ns["shift_logits_and_labels"](logits, labels)
    with step("drops last logit / first label"):
        expect_shape(sl, (2, 3, 5))
        expect_shape(slab, (2, 3))
        expect_allclose(slab, labels[:, 1:])


def test_0021_cross_entropy_loss(ns):
    logits = torch.randn(2, 3, 7)
    labels = torch.randint(0, 7, (2, 3))
    labels[0, 0] = -100
    import torch.nn.functional as F
    want = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1), ignore_index=-100)
    with step("matches F.cross_entropy with ignore_index"):
        expect_allclose(ns["cross_entropy_loss"](logits, labels), want)


def test_0022_adamw_update(ns):
    x = torch.tensor([0.0])
    m = torch.zeros(1)
    v = torch.zeros(1)
    for t in range(1, 401):
        grad = 2.0 * (x - 3.0)
        x, m, v = ns["adamw_update"](x, grad, m, v, t, lr=0.1, weight_decay=0.0)
    with step("AdamW minimizes (x-3)^2"):
        expect_true(abs(x.item() - 3.0) < 0.1, f"x={x.item()}")


def test_0023_linear_warmup_schedule(ns):
    with step("ramps to base_lr then holds"):
        expect_allclose(ns["linear_warmup_schedule"](50, 100, 1.0), 0.5)
        expect_allclose(ns["linear_warmup_schedule"](100, 100, 1.0), 1.0)
        expect_allclose(ns["linear_warmup_schedule"](500, 100, 1.0), 1.0)


def test_0024_clip_grad_norm(ns):
    big = [torch.tensor([3.0, 4.0])]  # norm 5
    clipped = ns["clip_grad_norm"](big, 1.0)
    total = torch.sqrt(sum((g ** 2).sum() for g in clipped))
    with step("scales down to max_norm; leaves small grads alone"):
        expect_allclose(total, 1.0, atol=1e-3)
        small = [torch.tensor([0.1, 0.1])]
        expect_allclose(ns["clip_grad_norm"](small, 1.0)[0], small[0])


def test_0025_accumulate_gradients(ns):
    g1 = [torch.tensor([2.0, 4.0])]
    g2 = [torch.tensor([4.0, 8.0])]
    out = ns["accumulate_gradients"]([g1, g2])
    with step("averages the microbatch grads"):
        expect_allclose(out[0], [3.0, 6.0])


def test_0026_sft_train_step(ns):
    torch.manual_seed(0)
    model = _TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = _tiny_batch()
    first = ns["sft_train_step"](model, batch, opt)
    last = first
    for _ in range(40):
        last = ns["sft_train_step"](model, batch, opt)
    with step("training reduces the SFT loss"):
        expect_true(last < first - 0.1, f"loss should drop: first={first:.3f} last={last:.3f}")


def test_0027_evaluate_loss(ns):
    torch.manual_seed(0)
    model = _TinyLM()
    batches = [_tiny_batch(seed=1), _tiny_batch(seed=2)]
    val = ns["evaluate_loss"](model, batches)
    with step("returns a finite mean loss"):
        expect_true(np.isfinite(val), "loss must be finite")


# --------------------- Part 4 — LoRA Adapters ---------------------

def test_0028_lora_delta(ns):
    x = torch.randn(3, 8)
    a = torch.randn(4, 8)   # (r, in)
    b = torch.randn(6, 4)   # (out, r)
    out = ns["lora_delta"](x, a, b, alpha=8, r=4)
    with step("(alpha/r)*(x@A^T)@B^T, shape (3,6)"):
        expect_shape(out, (3, 6))
        expect_allclose(out, (8 / 4) * (x @ a.T) @ b.T)


def test_0029_lora_linear_forward(ns):
    x = torch.randn(3, 8)
    w = torch.randn(6, 8)
    bias = torch.randn(6)
    a, b = torch.randn(4, 8), torch.randn(6, 4)
    import torch.nn.functional as F
    out = ns["lora_linear_forward"](x, w, bias, a, b, 8, 4)
    with step("base linear + LoRA delta"):
        expect_allclose(out, F.linear(x, w, bias) + (8 / 4) * (x @ a.T) @ b.T)


def test_0030_init_lora_weights(ns):
    a, b = ns["init_lora_weights"](8, 6, 4, torch.Generator().manual_seed(0))
    with step("A small Gaussian (r,in); B zeros (out,r) -> zero initial delta"):
        expect_shape(a, (4, 8))
        expect_shape(b, (6, 4))
        expect_allclose(b, torch.zeros(6, 4))
        expect_allclose(ns["lora_delta"](torch.randn(2, 8), a, b, 8, 4), torch.zeros(2, 6))


def test_0031_freeze_base_params(ns):
    model = _TinyLM()
    ns["freeze_base_params"](model)
    with step("all params frozen"):
        expect_true(all(not p.requires_grad for p in model.parameters()), "should all be frozen")


def test_0032_count_trainable_params(ns):
    model = _TinyLM(vocab=10, d=4)
    full = ns["count_trainable_params"](model)
    ns["freeze_base_params"](model)
    with step("counts only requires_grad params"):
        expect_true(full > 0, "fresh model has trainable params")
        expect_eq(ns["count_trainable_params"](model), 0)


def test_0033_merge_lora(ns):
    x = torch.randn(3, 8)
    w = torch.randn(6, 8)
    a, b = torch.randn(4, 8), torch.randn(6, 4)
    import torch.nn.functional as F
    merged = ns["merge_lora"](w, a, b, 8, 4)
    with step("merged weight reproduces base+LoRA forward"):
        expect_shape(merged, (6, 8))
        expect_allclose(F.linear(x, merged), ns["lora_linear_forward"](x, w, None, a, b, 8, 4), atol=1e-4)


# --------------------- Part 5 — Reward Modeling ---------------------

def test_0034_build_synthetic_preference_dataset(ns):
    data = ns["build_synthetic_preference_dataset"](15)
    with step("n examples with prompt/chosen/rejected"):
        expect_eq(len(data), 15)
        expect_true(all(k in data[0] for k in ("prompt", "chosen", "rejected")), "missing keys")


def test_0035_format_preference(ns):
    out = ns["format_preference"]({"prompt": "Q?", "chosen": "good", "rejected": "bad"})
    with step("chosen/rejected texts share the prompt and append the response"):
        expect_true(out["chosen_text"].endswith("good"), "chosen text")
        expect_true(out["rejected_text"].endswith("bad"), "rejected text")
        expect_true("Q?" in out["chosen_text"], "prompt present")


def test_0036_reward_head_forward(ns):
    h = torch.randn(2, 3, 4)
    w = torch.randn(4)
    b = torch.tensor(0.5)
    out = ns["reward_head_forward"](h, w, b)
    with step("scalar reward per sequence from the last token"):
        expect_shape(out, (2,))
        expect_allclose(out, h[:, -1, :] @ w + b)


def test_0037_pairwise_reward_loss(ns):
    import torch.nn.functional as F
    c = torch.tensor([2.0, 1.0])
    r = torch.tensor([0.0, -1.0])
    with step("matches -logsigmoid(c-r); shrinks as the gap grows"):
        expect_allclose(ns["pairwise_reward_loss"](c, r), -F.logsigmoid(c - r).mean())
        big = ns["pairwise_reward_loss"](torch.tensor([10.0]), torch.tensor([-10.0]))
        expect_true(big.item() < 0.01, "huge margin -> ~0 loss")


def test_0038_reward_bce_loss(ns):
    import torch.nn.functional as F
    c = torch.tensor([2.0, 1.0])
    r = torch.tensor([0.0, -1.0])
    with step("pointwise BCE (chosen=1, rejected=0)"):
        expect_allclose(ns["reward_bce_loss"](c, r), (-F.logsigmoid(c) - F.logsigmoid(-r)).mean())


def test_0039_pairwise_accuracy(ns):
    with step("fraction with chosen > rejected"):
        expect_allclose(ns["pairwise_accuracy"](torch.tensor([3.0, 1.0]), torch.tensor([1.0, 2.0])), 0.5)


def test_0040_reward_train_step(ns):
    torch.manual_seed(0)
    rm = _TinyReward()
    opt = torch.optim.Adam(rm.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(1)
    batch = {"chosen": torch.randint(0, 20, (4, 5), generator=g),
             "rejected": torch.randint(0, 20, (4, 5), generator=g)}
    first = ns["reward_train_step"](rm, batch, opt)
    last = first
    for _ in range(60):
        last = ns["reward_train_step"](rm, batch, opt)
    with step("reward model learns to separate chosen from rejected"):
        expect_true(last < first - 0.1, f"loss should drop: first={first:.3f} last={last:.3f}")


# --------------------- Part 6 — PPO-Based RLHF ---------------------

def test_0041_sequence_logprob(ns):
    import torch.nn.functional as F
    logits = torch.randn(2, 4, 6)
    ids = torch.randint(0, 6, (2, 4))
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    want = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).sum(-1)
    with step("sum of realized-token log-probs, shape (B,)"):
        expect_shape(ns["sequence_logprob"](logits, ids), (2,))
        expect_allclose(ns["sequence_logprob"](logits, ids), want)


def test_0042_per_token_kl(ns):
    lp = torch.tensor([0.0, -1.0, -2.0])
    ref = torch.tensor([-0.5, -1.0, -1.0])
    with step("logp - logp_ref"):
        expect_allclose(ns["per_token_kl"](lp, ref), lp - ref)


def test_0043_compute_returns(ns):
    out = ns["compute_returns"](torch.tensor([0.0, 0.0, 1.0]), 0.9)
    with step("discounted returns-to-go"):
        expect_allclose(out, [0.81, 0.9, 1.0])


def test_0044_gae_advantages(ns):
    # rewards [1,1], values [0,0,0], gamma=lam=1 -> deltas [1,1] -> adv [2,1]
    adv = ns["gae_advantages"](torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0, 0.0]), 1.0, 1.0)
    with step("GAE recursion"):
        expect_allclose(adv, [2.0, 1.0])


def test_0045_policy_ratio(ns):
    lp = torch.tensor([0.0, -1.0])
    old = torch.tensor([-1.0, -1.0])
    with step("exp(logp - logp_old)"):
        expect_allclose(ns["policy_ratio"](lp, old), torch.exp(lp - old))


def test_0046_clipped_surrogate(ns):
    with step("clips the ratio on the advantage"):
        # ratio 1.5, adv +1, eps .2 -> min(1.5, 1.2) = 1.2
        expect_allclose(ns["clipped_surrogate"](torch.tensor([1.5]), torch.tensor([1.0]), 0.2), 1.2)
        # ratio 1.5, adv -1 -> min(-1.5, -1.2) = -1.5
        expect_allclose(ns["clipped_surrogate"](torch.tensor([1.5]), torch.tensor([-1.0]), 0.2), -1.5)


def test_0047_value_function_loss(ns):
    v = torch.tensor([1.0, 2.0])
    r = torch.tensor([0.0, 0.0])
    with step("0.5 * mean((v-r)^2)"):
        expect_allclose(ns["value_function_loss"](v, r), 0.5 * ((v - r) ** 2).mean())


def test_0048_entropy_bonus(ns):
    logits = torch.zeros(1, 4)  # uniform -> entropy log(4)
    with step("entropy of a uniform distribution is log(V)"):
        expect_allclose(ns["entropy_bonus"](logits), torch.log(torch.tensor(4.0)))


def test_0049_ppo_loss(ns):
    with step("-surrogate + vf_coef*vloss - ent_coef*entropy"):
        out = ns["ppo_loss"](torch.tensor(2.0), torch.tensor(1.0), torch.tensor(0.5), 0.5, 0.01)
        expect_allclose(out, -2.0 + 0.5 * 1.0 - 0.01 * 0.5)


def test_0050_kl_penalized_reward(ns):
    rewards = torch.tensor([1.0, 1.0])
    lp = torch.tensor([0.0, -1.0])
    ref = torch.tensor([-1.0, -1.0])
    with step("reward minus kl_coef * (logp - logp_ref)"):
        expect_allclose(ns["kl_penalized_reward"](rewards, lp, ref, 0.2), rewards - 0.2 * (lp - ref))


def test_0051_batch_sequence_logprob(ns):
    torch.manual_seed(0)
    model = _TinyLM()
    ids = torch.randint(0, 20, (3, 5))
    with step("matches sequence_logprob on the model's logits"):
        want = ns["sequence_logprob"](model(input_ids=ids).logits, ids)
        expect_allclose(ns["batch_sequence_logprob"](model, ids), want, atol=1e-4)
