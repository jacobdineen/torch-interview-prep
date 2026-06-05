"""
Step 0012: layer_forward

Part 4 — Neural Network
Apply every neuron in the layer to the same input list x, returning their outputs.
"""
import math  # noqa: F401


def layer_forward(layer, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
