"""
Step 0007: block_exp_scores

Part 2 — Online Softmax
Unnormalized block probabilities: exp of the block scores shifted by the new running max.
"""
import numpy as np  # noqa: F401


def block_exp_scores(S_block, m_new):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
