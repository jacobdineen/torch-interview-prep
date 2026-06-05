"""
Step 0004: v_sub

Part 1 — Scalar Arithmetic
Return a Value holding the difference a minus b, propagating gradient with opposite signs.
"""
import math  # noqa: F401


def v_sub(a, b):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
