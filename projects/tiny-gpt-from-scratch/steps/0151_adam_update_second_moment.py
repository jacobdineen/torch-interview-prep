"""
Step 0151: adam_update_second_moment

Part 8 — Adam, Training Loop, and Generation
EMA of the squared gradient: v = beta2*v + (1-beta2)*grad**2.
"""
import numpy as np  # noqa: F401


def adam_update_second_moment(v, grad, beta2):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
