"""
Step 0013: lstm_forward

Part 3 — LSTM
Unroll the LSTM cell over the sequence axis, returning all hidden states and the final h and c.
"""
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
