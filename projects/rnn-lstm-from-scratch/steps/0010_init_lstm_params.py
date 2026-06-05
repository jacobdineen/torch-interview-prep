"""
Step 0010: init_lstm_params

Part 3 — LSTM
Initialize LSTM weight/bias tensors (4H gate block ordered [i,f,o,g]) as small random requires_grad params.

Return a dict with keys "Wx" (vocab_size,4*hidden), "Wh" (hidden,4*hidden), "b" (4*hidden,), "Why" (hidden,vocab_size), "by" (vocab_size,); the 4*hidden axis is the concatenated [i,f,o,g] gate block; weights small random (~0.01*randn) seeded by seed, biases zero, all float32 with requires_grad=True.

Conventions: inputs X are one-hot, shape (B, S, V); targets Y are (B, S) longs; hidden size H.
Parameters are a dict of tensors with requires_grad=True; a linear is x @ W + b with W (in, out).
Training uses torch autograd (loss.backward() then a manual SGD update under torch.no_grad()).
The RNN and LSTM share one train/sample path — you pass rnn_logits or lstm_logits as forward_fn.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py rnn-lstm-from-scratch` (or the outline drawer) to see all signatures."""
import torch  # noqa: F401


def init_lstm_params(vocab_size, hidden, seed=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
