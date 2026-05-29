"""
Step 0135: residual_forward

Part 7 — FFN, Blocks, and Full Model
Residual connection: x + sublayer(x).
"""
import numpy as np  # noqa: F401


def residual_forward(x, sublayer_out):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
