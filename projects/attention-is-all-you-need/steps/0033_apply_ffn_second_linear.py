"""
Step 0033: apply_ffn_second_linear

Part 5 — Feed-Forward, LayerNorm, and Dropout
Project the inner feed-forward activations back down to the model dimension.
"""
import torch  # noqa: F401


def apply_ffn_second_linear(h, ffn_params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
