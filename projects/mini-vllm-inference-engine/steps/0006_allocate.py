"""
Step 0006: allocate

Part 2 — PagedAttention: Paging
Pop ceil(n_tokens/block_size) blocks from the front of the free-list, or None if OOM.
"""
import numpy as np  # noqa: F401


def allocate(mgr, n_tokens):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
