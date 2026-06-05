"""
Step 0011: paged_attention

Part 3 — PagedAttention: Paged Compute
Compute attention of q over the paged KV pool, equivalent to contiguous kv_attention.
"""
import numpy as np  # noqa: F401


def paged_attention(q, k_pool, v_pool, block_table, num_tokens, block_size, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
