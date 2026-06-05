"""
Step 0008: build_topo

Part 3 — Reverse-Mode Autodiff
Return the nodes of root's graph in depth-first postorder (children before parents, root last).
"""
import math  # noqa: F401


def build_topo(root):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
