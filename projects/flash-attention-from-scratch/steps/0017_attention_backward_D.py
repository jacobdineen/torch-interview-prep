"""
Step 0017: attention_backward_D

Part 5 — Flash Backward (Recomputation)
Per-row dot product of the upstream gradient and the attention output.
"""
import numpy as np  # noqa: F401


def attention_backward_D(dO, O):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
