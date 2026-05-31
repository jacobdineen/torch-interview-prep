"""
Step 0054: greedy_agent_action

Part 8 — Agents and Evaluation
Greedy column from the network's policy (no search).
"""
import torch  # noqa: F401


def greedy_agent_action(net, board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
