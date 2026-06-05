"""
Step 0007: rnn_forward

Part 2 — Vanilla RNN
Unroll the RNN cell over the sequence axis starting from h0.

If h0 is None use a zeros(B,hidden) initial state. Apply rnn_cell_step at each t; return (H_seq, h_last) where H_seq stacks all hidden states along dim=1 with shape (B,S,hidden) and h_last is the final (B,hidden) state.

Conventions: inputs X are one-hot, shape (B, S, V); targets Y are (B, S) longs; hidden size H.
Parameters are a dict of tensors with requires_grad=True; a linear is x @ W + b with W (in, out).
Training uses torch autograd (loss.backward() then a manual SGD update under torch.no_grad()).
The RNN and LSTM share one train/sample path — you pass rnn_logits or lstm_logits as forward_fn.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py rnn-lstm-from-scratch` (or the outline drawer) to see all signatures."""
import torch  # noqa: F401


def rnn_forward(X_onehot, h0, params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
