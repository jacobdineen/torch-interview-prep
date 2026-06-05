"""
Step 0015: masked_attention_reference

Part 4 — Causal Flash Attention
Compute full causal self-attention by masking future keys, softmaxing each row, and weighting V.
"""
import numpy as np  # noqa: F401


def masked_attention_reference(Q, K, V, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
