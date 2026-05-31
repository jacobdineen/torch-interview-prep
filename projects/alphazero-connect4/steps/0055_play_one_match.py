"""
Step 0055: play_one_match

Part 8 — Agents and Evaluation
Play a game: agent_a is player +1, agent_b is player -1. Each agent is a
callable(board, player) -> action. Returns the winner (+1/-1/0).
"""
import torch  # noqa: F401


def play_one_match(agent_a, agent_b, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
