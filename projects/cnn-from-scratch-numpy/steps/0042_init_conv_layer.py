"""
Step 0042: init_conv_layer

Part 5 — Assembling LeNet
Init a conv layer dict with He weights (out_ch,in_ch,k,k), zero bias, stride 1, same-pad.
"""
import numpy as np  # noqa: F401


def init_conv_layer(in_ch, out_ch, k, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
