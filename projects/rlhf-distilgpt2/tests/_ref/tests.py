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
