"""
Step 0002: v_mul

Part 1 — Scalar Arithmetic
Return a Value holding the product of a and b, with the product-rule gradient.
"""
import math  # noqa: F401


def v_mul(a, b):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
