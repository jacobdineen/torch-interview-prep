"""
Step 0005: init_rnn_params

Part 2 — Vanilla RNN
Create the trainable parameter dict for a vanilla RNN.
"""
import torch  # noqa: F401


def init_rnn_params(vocab_size, hidden, seed=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
