"""
Step 0017: sgd_step

Part 5 — Training
Update each parameter's data in place by a gradient-descent step of size lr.
"""
import math  # noqa: F401


def sgd_step(params, lr):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
