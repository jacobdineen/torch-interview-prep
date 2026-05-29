"""
Step 0139: transformer_block_backward

Part 7 — FFN, Blocks, and Full Model
Backward through a pre-LN Transformer block. Returns (dx, grads) where grads
mirrors the block dict (ln1, attn, ln2, ffn).
"""
import numpy as np  # noqa: F401


def transformer_block_backward(dout, x, block, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
