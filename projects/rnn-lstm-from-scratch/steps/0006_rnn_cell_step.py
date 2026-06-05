"""
Step 0006: rnn_cell_step

Part 2 — Vanilla RNN
Advance the RNN hidden state by one time step.
"""
import torch  # noqa: F401


def rnn_cell_step(x_t, h_prev, params):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
