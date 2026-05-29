"""
Step 0034: read_text_file

Part 3 — Data Pipeline and Bigram Baseline
Return the corpus text. Here the corpus is passed in directly as a string,
so this is effectively identity (real datasets would read from ``path``).
"""
import numpy as np  # noqa: F401


def read_text_file(path_or_text):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
