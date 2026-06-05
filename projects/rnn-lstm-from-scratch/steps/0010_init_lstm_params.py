"""
Step 0010: init_lstm_params

Part 3 — LSTM
Initialize LSTM weight/bias tensors (4H gate block ordered [i,f,o,g]) as small random requires_grad params.
"""
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
