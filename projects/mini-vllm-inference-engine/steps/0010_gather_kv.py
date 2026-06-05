"""
Step 0010: gather_kv

Part 3 — PagedAttention: Paged Compute
Read the first num_tokens KV slots from the pools into logically-contiguous K,V arrays.
"""
import numpy as np  # noqa: F401


def gather_kv(k_pool, v_pool, block_table, num_tokens, block_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
