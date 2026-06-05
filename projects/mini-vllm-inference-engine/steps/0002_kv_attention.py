"""
Step 0002: kv_attention

Part 1 — KV-Cache Decode
Compute scaled-dot-product attention of one query over cached keys/values.
"""
import numpy as np  # noqa: F401


def kv_attention(q, K, V, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
