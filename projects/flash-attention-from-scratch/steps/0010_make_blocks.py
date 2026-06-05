"""
Step 0010: make_blocks

Part 3 — Flash Forward (Tiled)
Contiguous (start, end) index ranges that tile 0..n; the last may be shorter.

Conventions: single-head, no batch. Q (N,d), K (M,d), V (M,d), output O (N,d); scores S (N,M);
per-row stats m, l, L, D are shape (N,). `scale` is the scalar the caller passes (1/sqrt(d)).
The flash forward tiles over key/value blocks and keeps an UNNORMALIZED running output that is
divided by the running denominator l only at the end.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py flash-attention-from-scratch` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def make_blocks(n, block_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
