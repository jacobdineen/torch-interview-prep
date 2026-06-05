"""
Step 0007: rnn_forward

Part 2 — Vanilla RNN
Unroll the RNN cell over the sequence axis starting from h0.
"""
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
