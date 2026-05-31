"""Reference implementations for rlhf-distilgpt2 (HIDDEN).

A from-scratch RLHF pipeline on distilgpt2 in PyTorch + HuggingFace transformers.
Conventions:
  * Pure-tensor steps (decoding filters, losses, PPO/GAE, preference losses)
    operate on plain torch tensors and are graded by value.
  * Model-dependent steps use the real distilgpt2 (small; downloaded once).

Each step's test grades the user's function by swapping it in over this reference
namespace, so steps are checked in isolation.
"""
import torch
import torch.nn.functional as F

MODEL_NAME = "distilgpt2"


# ============== Part 1 — Model Setup and Decoding Strategies ==============

def load_distilgpt2_tokenizer():
    """Load the distilgpt2 tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_distilgpt2_model():
    """Load the distilgpt2 causal-LM model."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def set_pad_token_to_eos(tokenizer):
    """GPT-2 has no pad token; reuse the EOS token for padding. Returns the tokenizer."""
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_and_decode(model, tokenizer, prompt, max_new_tokens):
    """Greedy-generate a continuation of ``prompt`` and return the decoded string."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def greedy_decode(logits):
    """Greedy next-token id(s): argmax over the vocabulary axis."""
    return torch.argmax(logits, dim=-1)


def sample_with_temperature(logits, temperature, generator=None):
    """Sample a token id from softmax(logits / temperature)."""
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def top_k_filter(logits, k):
    """Keep the top-k logits per row; set the rest to -inf."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability reaches ``p``; set the rest to -inf."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove_sorted = (cum - torch.softmax(sorted_logits, dim=-1)) > p  # keep the token that crosses p
    remove = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))
