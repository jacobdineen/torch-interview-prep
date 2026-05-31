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
