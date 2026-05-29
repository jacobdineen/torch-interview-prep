"""
Step 0145: full_model_forward

Part 7 — FFN, Blocks, and Full Model
Full GPT forward: embeddings -> blocks -> final LN -> LM head. Returns logits.
"""
import numpy as np  # noqa: F401


def full_model_forward(params, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
