"""
Step 0028: split_qkv_into_heads

Part 4 — Multi-Head Attention
Split each of query, key, and value into multiple attention heads.
"""
import torch  # noqa: F401


def split_qkv_into_heads(Q, K, V, n_heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
