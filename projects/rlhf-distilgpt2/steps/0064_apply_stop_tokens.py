"""
Step 0064: apply_stop_tokens

Part 8 — Evaluation and Chat Interface
Truncate ``text`` at the earliest occurrence of any stop string.
"""
import torch  # noqa: F401


def apply_stop_tokens(text, stop_strings):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
