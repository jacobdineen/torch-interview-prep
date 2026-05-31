"""
Step 0051: self_play_iteration

Part 7 — Iterated Training Loop
One AlphaZero iteration: generate self-play data, then train one epoch on it.
Returns (mean_loss, data).
"""
import torch  # noqa: F401


def self_play_iteration(net, optimizer, n_games, n_simulations, c_puct, temperature, batch_size, weight_decay, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
