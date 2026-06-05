"""
Step 0027: project_to_query_key_value

Part 4 — Multi-Head Attention
Linearly project inputs into query, key, and value representations.
"""
import torch  # noqa: F401


def project_to_query_key_value(x_q, x_kv, attn_params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
