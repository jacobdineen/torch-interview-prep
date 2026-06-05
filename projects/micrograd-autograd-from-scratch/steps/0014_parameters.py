"""
Step 0014: parameters

Part 5 — Training
Return a flat list of all weight and bias Values across every neuron in every layer.
"""
import math  # noqa: F401


def parameters(mlp):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
