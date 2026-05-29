"""
Step 0165: generation_loop_for_n_steps

Part 8 — Adam, Training Loop, and Generation
Autoregressively generate ``n_new_tokens`` tokens with temperature + top-k.
"""
import numpy as np  # noqa: F401


def generation_loop_for_n_steps(params, prompt_ids, n_new_tokens, block_size, temperature, top_k, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
