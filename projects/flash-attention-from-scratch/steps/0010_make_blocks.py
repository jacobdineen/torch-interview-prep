"""
Step 0010: make_blocks

Part 3 — Flash Forward (Tiled)
Contiguous (start, end) index ranges that tile 0..n; the last may be shorter.
"""
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
