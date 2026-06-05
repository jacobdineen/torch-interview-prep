"""
Step 0007: free

Part 2 — PagedAttention: Paging
Return every block id in the table to the back of the free-list in order.
"""
import numpy as np  # noqa: F401


def free(mgr, block_table):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
