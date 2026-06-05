"""
Step 0006: update_running_max

Part 2 — Online Softmax
Elementwise running maximum of the previous max and the new block's row max.
"""
import numpy as np  # noqa: F401


def update_running_max(m_prev, m_block):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
