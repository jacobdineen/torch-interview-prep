# Tiny GPT From Scratch

A character-level GPT built up from raw NumPy — no autograd, no framework. You derive every forward and backward pass by hand, from the tokenizer through multi-head self-attention to a full pre-LN Transformer trained with Adam.

**166 steps across 8 parts.** Each step is one function (or class) in `steps/NNNN_<fn>.py`, graded in isolation against a hidden reference.

## Parts

| # | Part | Steps | Focus |
|---|------|-------|-------|
| 1 | Tokenizer | 7 | Build a tiny character-level tokenizer with vocab, stoi/itos, and encode/decode helpers. |
| 2 | NumPy and Softmax Foundations | 26 | Get fluent with NumPy arrays, indexing, broadcasting, reductions, and numerically stable softmax. |
| 3 | Data Pipeline and Bigram Baseline | 23 | Load the corpus, build batched (X, Y) sequences, and train a counting-based bigram model. |
| 4 | Single-Layer Neural Bigram | 17 | Replace the count table with a learned weight matrix; derive cross-entropy, gradients, SGD. |
| 5 | Layer Primitives and Backprop | 18 | Forward and backward for linear, bias, ReLU, softmax+CE, and LayerNorm building blocks. |
| 6 | Embeddings and Self-Attention | 39 | Token/positional embeddings, then masked single- and multi-head self-attention with backward. |
| 7 | FFN, Blocks, and Full Model | 16 | Feed-forward, residuals, pre-LN Transformer blocks, and the complete GPT forward/backward. |
| 8 | Adam, Training Loop, and Generation | 20 | Adam, the full training/validation loop, and sampling with temperature and top-k decoding. |

## Working through it

Run from the repo root. The CLI tracks progress and unlocks the next step as you pass each one.

```bash
uv run python projects.py tiny-gpt-from-scratch              # parts + steps, [x]/[ ] solved
uv run python projects.py tiny-gpt-from-scratch --next       # jump to the next unsolved step
uv run python projects.py tiny-gpt-from-scratch <id>         # check one step, e.g. 0001
uv run python projects.py tiny-gpt-from-scratch <id> --explain   # hint / explanation
```

In Neovim, open a step file and use the `<leader>p` practice maps (run / next / hint / explain / solution) — they route to this project automatically.

## End-to-end scaffold

Once enough steps pass, assemble everything and run the full demo:

```bash
uv run python projects.py tiny-gpt-from-scratch --scaffold
```

Trains a tiny GPT on a toy corpus; validation loss falls from ~3.3 to ~0.4 and it samples coherent text.
