"""
Step 0033: expand_node

Part 4 — PUCT Monte Carlo Tree Search
Create a child for every legal move, carrying its prior. Marks node expanded.
"""
import torch  # noqa: F401


def expand_node(node, priors):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
