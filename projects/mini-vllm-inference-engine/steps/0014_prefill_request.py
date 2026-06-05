"""
Step 0014: prefill_request

Part 4 — Continuous Batching
Process all prompt tokens into the paged pool and return the running-request dict.
"""
import numpy as np  # noqa: F401


def prefill_request(model, req_id, prompt_ids, block_table, k_pool, v_pool, block_size, scale, max_new):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
