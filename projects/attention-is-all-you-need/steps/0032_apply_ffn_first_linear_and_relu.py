"""
Step 0032: apply_ffn_first_linear_and_relu

Part 5 — Feed-Forward, LayerNorm, and Dropout
Project the input up to the feed-forward inner dimension and apply a ReLU nonlinearity.
"""
import torch  # noqa: F401


def apply_ffn_first_linear_and_relu(x, ffn_params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
