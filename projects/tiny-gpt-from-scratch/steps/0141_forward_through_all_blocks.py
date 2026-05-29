"""
Step 0141: forward_through_all_blocks

Part 7 — FFN, Blocks, and Full Model
Run the input through every Transformer block in order.
"""
import numpy as np  # noqa: F401


def forward_through_all_blocks(x, blocks, mask):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
