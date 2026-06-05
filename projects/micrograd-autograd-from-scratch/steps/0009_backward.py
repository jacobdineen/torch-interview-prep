"""
Step 0009: backward

Part 3 — Reverse-Mode Autodiff
Seed root's grad to 1.0 and run each node's _backward in reverse topological order.
"""
import math  # noqa: F401


def backward(root):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
