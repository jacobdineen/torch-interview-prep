# Build an RNN and LSTM from Scratch

Build a character-level language model from the ground up in PyTorch: a vanilla
**RNN** cell and a gated **LSTM** cell, unrolled over sequences, trained with
torch autograd, and sampled autoregressively. **21 steps, 5 parts.**

| Part | Focus |
|------|-------|
| 1 | Character data: vocab, encoding, next-char windows, one-hot |
| 2 | Vanilla RNN: cell step, unroll, logits |
| 3 | LSTM: the i/f/o/g gates, cell step, unroll, logits |
| 4 | Sequence cross-entropy, SGD step (autograd), training loop |
| 5 | Temperature sampling, autoregressive generation, perplexity |

The RNN and LSTM share one training/sampling path — you pass `rnn_logits` or
`lstm_logits` as the forward fn.

```bash
uv run python projects.py rnn-lstm-from-scratch            # parts + steps
uv run python projects.py rnn-lstm-from-scratch --scaffold # train RNN+LSTM, sample text
```
