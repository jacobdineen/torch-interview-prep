"""
Step 0018: radix_longest_prefix

Part 5 — Prefix Cache & Sampling
Return the length of the longest leading run of tokens already cached as a path.
"""
import numpy as np  # noqa: F401


def radix_longest_prefix(root, token_ids):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
