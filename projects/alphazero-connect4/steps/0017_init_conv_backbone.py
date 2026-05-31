"""
Step 0017: init_conv_backbone

Part 2 — Board Encoding and Policy-Value Network
A small conv backbone preserving the 6x7 spatial size.
"""
import torch  # noqa: F401


def init_conv_backbone(in_channels, hidden):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
