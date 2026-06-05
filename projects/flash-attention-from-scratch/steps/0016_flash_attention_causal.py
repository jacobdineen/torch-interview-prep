"""
Step 0016: flash_attention_causal

Part 4 — Causal Flash Attention
Compute causal self-attention by tiling over key blocks, masking future keys per block before each online-softmax step.
"""
import numpy as np  # noqa: F401


def flash_attention_causal(Q, K, V, scale, block_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
