"""
Step 0042: generate_self_play_batch

Part 5 — Self-Play Data Generation
Play ``n_games`` self-play games and return all (state, policy, value) tuples.
"""
import torch  # noqa: F401


def generate_self_play_batch(net, n_games, n_simulations, c_puct, temperature, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
