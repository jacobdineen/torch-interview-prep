"""
Step 0003: decode_step

Part 1 — KV-Cache Decode
Process one token: append its k/v, attend over the cache, return logits and the new k,v.
"""
import numpy as np  # noqa: F401


def decode_step(model, token_id, K_cache, V_cache, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
