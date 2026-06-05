"""
Step 0005: correction

Part 2 — Online Softmax
Rescaling factor exp(m_old - m_new) for old stats when the running max grows (m_old=-inf -> 0).
"""
import numpy as np  # noqa: F401


def correction(m_old, m_new):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
