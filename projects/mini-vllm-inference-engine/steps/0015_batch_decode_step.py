"""
Step 0015: batch_decode_step

Part 4 — Continuous Batching
Advance every running request by one greedy token; return those that just finished.
"""
import numpy as np  # noqa: F401


def batch_decode_step(model, running, k_pool, v_pool, block_size, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
