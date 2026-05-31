"""
Step 0028: node_q_value

Part 4 — PUCT Monte Carlo Tree Search
Mean value of a node from its own player's perspective (0 if unvisited).
"""
import torch  # noqa: F401


def node_q_value(node):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
