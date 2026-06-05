"""
Step 0016: flash_attention_causal

Part 4 — Causal Flash Attention
Compute causal self-attention by tiling over key blocks, masking future keys per block before each online-softmax step.

Return O ONLY — the finalized (N, d) output. Unlike flash_attention_forward, do NOT also return L. Tile over key blocks (q_start=0, k_start=ks) and mask each S_block with causal_block_mask before online_softmax_step.

Conventions: single-head, no batch. Q (N,d), K (M,d), V (M,d), output O (N,d); scores S (N,M);
per-row stats m, l, L, D are shape (N,). `scale` is the scalar the caller passes (1/sqrt(d)).
The flash forward tiles over key/value blocks and keeps an UNNORMALIZED running output that is
divided by the running denominator l only at the end.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py flash-attention-from-scratch` (or the outline drawer) to see all signatures."""
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
