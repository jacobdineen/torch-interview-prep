# Attention Is All You Need: Build the Transformer From Scratch

Reimplement the original encoder-decoder Transformer (Vaswani et al., 2017) end to
end in PyTorch — from token vocabularies and sinusoidal positional encodings through
multi-head attention, label smoothing, the Noam learning-rate schedule, a from-scratch
Adam optimizer, and beam search. By the end you have a working seq2seq Transformer
training loop and inference pipeline assembled from first principles.

**79 steps across 11 parts.** Each step is one function in `steps/NNNN_<fn>.py`, solved
in isolation against a hidden reference and an independent test. Later steps call earlier
ones, so the pieces compose into a real model — the final steps actually train (loss falls)
and decode.

| Part | Focus |
|------|-------|
| 1 | Tokenization & batching |
| 2 | Embeddings & sinusoidal positional encoding |
| 3 | Masks & scaled dot-product attention (step by step) |
| 4 | Multi-head attention |
| 5 | Feed-forward, LayerNorm, residual add-&-norm, dropout |
| 6 | Encoder / decoder layers & the full forward pass |
| 7 | Raw parameter initialization |
| 8 | Teacher forcing, Noam schedule, label-smoothed KL loss |
| 9 | Adam optimizer from scratch |
| 10 | Training step & loop |
| 11 | Greedy decoding & length-penalized beam search |

## Working through it

```bash
uv run python projects.py attention-is-all-you-need            # parts + steps, [x]/[ ] solved
uv run python projects.py attention-is-all-you-need --next     # jump to the next unsolved step
uv run python projects.py attention-is-all-you-need 0022 --explain
uv run python projects.py attention-is-all-you-need --scaffold # end-to-end copy-task demo (once solved)
```
