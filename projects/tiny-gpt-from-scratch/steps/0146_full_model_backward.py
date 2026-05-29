"""
Step 0146: full_model_backward

Part 7 — FFN, Blocks, and Full Model
Full GPT backward from dlogits. Returns a grads dict mirroring ``params``
(tok_emb, pos_emb, blocks, ln_f, lm_head).
"""
import numpy as np  # noqa: F401


def full_model_backward(params, x, dlogits):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
