"""
Step 0144: lm_head_linear_forward

Part 7 — FFN, Blocks, and Full Model
Project hidden states to vocab logits: x @ W_lm + b_lm -> (..., vocab).
"""
import numpy as np  # noqa: F401


def lm_head_linear_forward(x, w_lm, b_lm):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
