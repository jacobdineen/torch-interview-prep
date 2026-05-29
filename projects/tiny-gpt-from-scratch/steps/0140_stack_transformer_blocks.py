"""
Step 0140: stack_transformer_blocks

Part 7 — FFN, Blocks, and Full Model
Create ``n_layers`` randomly-initialized pre-LN Transformer blocks.
"""
import numpy as np  # noqa: F401


def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
