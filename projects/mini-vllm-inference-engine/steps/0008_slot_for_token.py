"""
Step 0008: slot_for_token

Part 2 — PagedAttention: Paging
Map a logical token index to its (physical block id, in-block offset).
"""
import numpy as np  # noqa: F401


def slot_for_token(block_table, token_index, block_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
