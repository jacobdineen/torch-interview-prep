"""
Step 0009: online_softmax_step

Part 2 — Online Softmax
One fused flash step folding a key/value block into the running (max, denominator, unnormalized output).
"""
import numpy as np  # noqa: F401


def online_softmax_step(m_prev, l_prev, O_prev, S_block, V_block):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
