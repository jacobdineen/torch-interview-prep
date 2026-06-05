"""
Step 0008: update_normalizer

Part 2 — Online Softmax
Updated softmax denominator: rescaled old denominator plus the new block's exp-score row sums.
"""
import numpy as np  # noqa: F401


def update_normalizer(l_prev, m_prev, m_new, p_block):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
