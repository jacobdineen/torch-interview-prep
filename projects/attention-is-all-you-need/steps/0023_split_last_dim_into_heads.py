"""
Step 0023: split_last_dim_into_heads

Part 4 — Multi-Head Attention
Reshape the trailing model dimension into separate head and per-head feature dimensions.
"""
import torch  # noqa: F401


def split_last_dim_into_heads(x, n_heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
