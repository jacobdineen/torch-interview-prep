"""
Step 0009: rnn_logits

Part 2 — Vanilla RNN
Run the RNN from a zero initial state and produce logits.
"""
import torch  # noqa: F401


def rnn_logits(params, X_onehot):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
