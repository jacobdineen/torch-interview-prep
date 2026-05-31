"""
Step 0057: evaluate_against_random

Part 8 — Agents and Evaluation
Greedy network agent (as +1) vs a random opponent; return its win rate.
"""
import torch  # noqa: F401


def evaluate_against_random(net, n_games, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
