"""
Step 0052: train_loop

Part 7 — Iterated Training Loop
Alternate self-play and training for ``n_iterations``. Returns (net, losses).
"""
import torch  # noqa: F401


def train_loop(net, optimizer, n_iterations, n_games, n_simulations, c_puct, temperature, batch_size, weight_decay, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
