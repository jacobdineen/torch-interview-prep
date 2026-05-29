"""
Step 0072: train_neural_bigram_loop

Part 4 — Single-Layer Neural Bigram
Train the neural bigram for ``n_steps`` SGD steps. Returns (W, losses).
"""
import numpy as np  # noqa: F401


def train_neural_bigram_loop(w, data, n_steps, batch_size, lr, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
