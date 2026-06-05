"""
Step 0005: new_block_manager

Part 2 — PagedAttention: Paging
Create a paging manager with a FIFO free-list of physical block ids.
"""
import numpy as np  # noqa: F401


def new_block_manager(num_blocks, block_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
