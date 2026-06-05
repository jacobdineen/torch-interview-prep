"""
Step 0011: lstm_gates

Part 3 — LSTM
Compute the LSTM input, forget, output, and candidate gates for one time step.

Compute z = x_t @ params["Wx"] + h_prev @ params["Wh"] + params["b"], split z into four (B,hidden) blocks in order [i,f,o,g], and return (sigmoid(i), sigmoid(f), sigmoid(o), tanh(g)).

Conventions: inputs X are one-hot, shape (B, S, V); targets Y are (B, S) longs; hidden size H.
Parameters are a dict of tensors with requires_grad=True; a linear is x @ W + b with W (in, out).
Training uses torch autograd (loss.backward() then a manual SGD update under torch.no_grad()).
The RNN and LSTM share one train/sample path — you pass rnn_logits or lstm_logits as forward_fn.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py rnn-lstm-from-scratch` (or the outline drawer) to see all signatures."""
import torch  # noqa: F401


def lstm_gates(x_t, h_prev, params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
