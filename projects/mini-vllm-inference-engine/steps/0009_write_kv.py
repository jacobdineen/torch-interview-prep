"""
Step 0009: write_kv

Part 3 — PagedAttention: Paged Compute
Write key/value vectors into the paged pool at the logical token's physical slot.
"""
import numpy as np  # noqa: F401


def write_kv(k_pool, v_pool, block_table, token_index, block_size, k, v):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
