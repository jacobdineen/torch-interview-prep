"""
Problem 68b: beam=1 matches greedy decoding

(Split from parent problem 68: Problem 49: Beam Search Decoding)
"""
import math
import torch

def beam_search(step_fn, start_tokens, beam_size, max_steps, end_token, length_penalty=0.0):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
