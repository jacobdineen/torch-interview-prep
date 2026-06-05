"""
Step 0016: run_batched

Part 4 — Continuous Batching
Continuously batch (admit/prefill/decode) all requests; return {req_id: generated ids}.
"""
import numpy as np  # noqa: F401


def run_batched(model, requests, num_blocks, block_size, scale, max_new):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
