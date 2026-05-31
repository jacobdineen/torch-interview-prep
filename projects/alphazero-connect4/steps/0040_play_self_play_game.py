"""
Step 0040: play_self_play_game

Part 5 — Self-Play Data Generation
Play one MCTS self-play game. Returns (records, winner) where records are
(encoded_state, mcts_policy, player) tuples.
"""
import torch  # noqa: F401


def play_self_play_game(net, n_simulations, c_puct, temperature, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
