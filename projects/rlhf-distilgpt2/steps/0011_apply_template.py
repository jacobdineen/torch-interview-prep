"""
Step 0011: apply_template

Part 2 — SFT Data Pipeline
Wrap a formatted example in the instruction template. Returns
{'prompt_text', 'full_text'} (full = prompt + response).
"""
import torch  # noqa: F401


def apply_template(formatted):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
