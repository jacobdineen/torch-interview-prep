"""
Step 0075: compute_candidate_scores

Part 11 — Decoding and Beam Search
Combine each beam's running log-prob with every possible next token's log-prob.
"""
import torch  # noqa: F401


def compute_candidate_scores(beam_logprob_sums, next_log_probs):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
