"""
Step 0010: build_merge_ranks

Part 3 — Training the Merges
Map each merge pair to its position (rank) in the learned merge list.
"""
import numpy as np  # noqa: F401


def build_merge_ranks(merges):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
