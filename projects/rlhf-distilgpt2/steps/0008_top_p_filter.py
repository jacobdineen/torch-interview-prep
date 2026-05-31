"""
Step 0008: top_p_filter

Part 1 — Model Setup and Decoding Strategies
Nucleus filtering: keep the smallest set of tokens whose cumulative
probability reaches ``p``; set the rest to -inf.
"""
import torch  # noqa: F401


def top_p_filter(logits, p):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
