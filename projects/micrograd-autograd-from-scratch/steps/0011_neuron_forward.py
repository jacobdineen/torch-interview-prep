"""
Step 0011: neuron_forward

Part 4 — Neural Network
Compute tanh of the weighted sum of inputs x plus the neuron's bias.
"""
import math  # noqa: F401


def neuron_forward(neuron, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
