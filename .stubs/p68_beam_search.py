"""
Problem 49: Beam Search Decoding

You're given a step function that produces the next-token log-probabilities given a
prefix. Implement beam search.

  - beam_search(step_fn, start_tokens, beam_size, max_steps, end_token, length_penalty=0.0)

        step_fn(prefix): (T,) LongTensor -> (V,) FloatTensor of log-probabilities for
                        the next token. (Tests will provide a deterministic step_fn.)

        start_tokens: 1D LongTensor of length T0, the shared prefix.

        Maintain `beam_size` live beams.
        At each step, expand every live beam by every token. Keep the top
        `beam_size` by total log-prob across all (beam, token) combinations.
        Beams that emit `end_token` are moved to `finished`.
        Stop when all beams have finished or max_steps is reached.

        Score = total log-prob / (length ** length_penalty).
        Return the FINISHED beam with the best score:
            (tokens: LongTensor, score: float).

The step_fn used by the tests is purely deterministic, so the answer is exact.
"""

import math
import torch

def beam_search(step_fn, start_tokens, beam_size, max_steps, end_token, length_penalty=0.0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
