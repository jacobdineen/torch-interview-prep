"""
Step 0014: lstm_logits

Part 3 — LSTM
Run the LSTM from zero initial states and project the hidden sequence to vocabulary logits.
"""
import torch  # noqa: F401


def lstm_logits(params, X_onehot):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
