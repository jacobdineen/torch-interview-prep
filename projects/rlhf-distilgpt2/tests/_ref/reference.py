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


# ============== Part 2 — SFT Data Pipeline ==============

_PROMPT_TEMPLATE = "### Instruction:\n{}\n### Response:\n"


def build_synthetic_instruction_dataset(n):
    """A deterministic synthetic instruction dataset (simple arithmetic Q&A)."""
    data = []
    for i in range(n):
        a, b = i % 10, (i * 3) % 10
        data.append({"instruction": f"What is {a} plus {b}?", "response": f"{a + b}"})
    return data


def format_example(example):
    """Normalize a raw example to {'prompt', 'response'} strings."""
    return {"prompt": example["instruction"], "response": example["response"]}


def apply_template(formatted):
    """Wrap a formatted example in the instruction template. Returns
    {'prompt_text', 'full_text'} (full = prompt + response)."""
    prompt_text = _PROMPT_TEMPLATE.format(formatted["prompt"])
    return {"prompt_text": prompt_text, "full_text": prompt_text + formatted["response"]}


def tokenize_example(templated, tokenizer):
    """Tokenize the full text; record how many tokens belong to the prompt.
    Returns {'input_ids', 'prompt_len'}."""
    prompt_ids = tokenizer(templated["prompt_text"]).input_ids
    full_ids = tokenizer(templated["full_text"]).input_ids
    return {"input_ids": full_ids, "prompt_len": len(prompt_ids)}


def build_labels(input_ids):
    """Labels for causal LM are a copy of the input ids (loss shifts them)."""
    return list(input_ids)


def mask_prompt_labels(labels, prompt_len, ignore_index=-100):
    """Set the prompt-token labels to ignore_index so loss only covers the response."""
    labels = list(labels)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = ignore_index
    return labels


def pad_batch(sequences, pad_value):
    """Right-pad a list of variable-length id sequences to a (batch, maxlen) tensor."""
    maxlen = max(len(s) for s in sequences)
    rows = [list(s) + [pad_value] * (maxlen - len(s)) for s in sequences]
    return torch.tensor(rows, dtype=torch.long)


def make_attention_mask(input_ids, pad_id):
    """1 for real tokens, 0 for padding."""
    return (input_ids != pad_id).long()


def collate_lm_batch(tokenized_examples, pad_id, ignore_index=-100):
    """Pad a batch of tokenized examples into input_ids / labels / attention_mask
    tensors, masking prompt tokens and padding in the labels."""
    input_ids = pad_batch([e["input_ids"] for e in tokenized_examples], pad_id)
    attention_mask = make_attention_mask(input_ids, pad_id)
    label_lists = [mask_prompt_labels(build_labels(e["input_ids"]), e["prompt_len"], ignore_index)
                   for e in tokenized_examples]
    labels = pad_batch(label_lists, ignore_index)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def iterate_minibatches(data, batch_size, shuffle=False, generator=None):
    """Split ``data`` into minibatches (optionally shuffled). Returns a list of batches."""
    idx = (torch.randperm(len(data), generator=generator).tolist() if shuffle
           else list(range(len(data))))
    return [[data[j] for j in idx[i:i + batch_size]] for i in range(0, len(data), batch_size)]


def train_val_split(data, val_fraction):
    """Split into (train, val); the last ``val_fraction`` of examples are validation."""
    n_val = int(len(data) * val_fraction)
    return data[:len(data) - n_val], data[len(data) - n_val:]


# ============== Part 3 — SFT Training Loop ==============

def shift_logits_and_labels(logits, labels):
    """Align next-token prediction: drop the last logit and the first label.
    (B,T,V),(B,T) -> (B,T-1,V),(B,T-1)."""
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def cross_entropy_loss(logits, labels, ignore_index=-100):
    """Mean cross-entropy over all positions, ignoring ``ignore_index`` labels."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=ignore_index)


def adamw_update(param, grad, m, v, t, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    """One AdamW step (decoupled weight decay) for one tensor. Returns (param, m, v)."""
    b1, b2 = betas
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad ** 2
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    param = param - lr * (mhat / (torch.sqrt(vhat) + eps) + weight_decay * param)
    return param, m, v


def linear_warmup_schedule(step, warmup_steps, base_lr):
    """Linearly ramp the LR from 0 to base_lr over warmup_steps, then hold."""
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, step / warmup_steps)


def clip_grad_norm(grads, max_norm):
    """Scale a list of gradient tensors so their global L2 norm is at most max_norm."""
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    scale = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return [g * scale for g in grads]


def accumulate_gradients(microbatch_grads):
    """Average a list of per-microbatch gradient lists into one gradient list."""
    n = len(microbatch_grads)
    return [sum(mb[i] for mb in microbatch_grads) / n for i in range(len(microbatch_grads[0]))]


def sft_train_step(model, batch, optimizer):
    """One supervised fine-tuning step (shifted next-token loss). Returns the loss value."""
    model.train()
    dev = next(model.parameters()).device
    out = model(input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev))
    logits, labels = shift_logits_and_labels(out.logits, batch["labels"].to(dev))
    loss = cross_entropy_loss(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_loss(model, batches):
    """Mean SFT loss over a list of batches (no gradient)."""
    model.eval()
    dev = next(model.parameters()).device
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            out = model(input_ids=batch["input_ids"].to(dev),
                        attention_mask=batch["attention_mask"].to(dev))
            logits, labels = shift_logits_and_labels(out.logits, batch["labels"].to(dev))
            total += cross_entropy_loss(logits, labels).item()
            n += 1
    return total / n
