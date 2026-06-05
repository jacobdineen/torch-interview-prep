"""
Step 0013: lstm_forward

Part 3 — LSTM
Unroll the LSTM cell over the sequence axis, returning all hidden states and the final h and c.

If h0 or c0 is None use a zeros(B,hidden) initial state for it. Return (H_seq, h_last, c_last) where H_seq stacks all hidden states along dim=1 with shape (B,S,hidden) and h_last,c_last are the final (B,hidden) states.

Conventions: inputs X are one-hot, shape (B, S, V); targets Y are (B, S) longs; hidden size H.
Parameters are a dict of tensors with requires_grad=True; a linear is x @ W + b with W (in, out).
Training uses torch autograd (loss.backward() then a manual SGD update under torch.no_grad()).
The RNN and LSTM share one train/sample path — you pass rnn_logits or lstm_logits as forward_fn.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py rnn-lstm-from-scratch` (or the outline drawer) to see all signatures."""
import torch  # noqa: F401


def lstm_forward(X_onehot, h0, c0, params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
