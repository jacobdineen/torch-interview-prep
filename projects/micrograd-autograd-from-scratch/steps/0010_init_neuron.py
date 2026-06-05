"""
Step 0010: init_neuron

Part 4 — Neural Network
Create a neuron as a dict of nin random weights in [-1,1] and a zero bias.
"""
import math  # noqa: F401


def init_neuron(nin, seed=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
