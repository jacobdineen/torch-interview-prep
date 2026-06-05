"""
Step 0004: greedy_generate

Part 1 — KV-Cache Decode
Greedily decode max_new tokens with a contiguous KV cache (the canonical ground truth).
"""
import numpy as np  # noqa: F401


def greedy_generate(model, prompt_ids, max_new, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
