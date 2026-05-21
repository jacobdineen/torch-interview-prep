"""
Problem 69: Greedy Decoding With a KV Cache

Tie generation to incremental attention. You're given a `step_fn` that takes the
current input tokens and a cache and returns next-token logits plus an updated
cache:

    step_fn(input_ids: LongTensor (B, T_step), cache: dict|None)
        -> (logits: FloatTensor (B, T_step, V), new_cache: dict)

Implement:

  - greedy_generate(step_fn, prompt_ids, max_new_tokens, eos_token=None)
        prompt_ids: (B, T_prompt) long. Generates tokens autoregressively, ALWAYS
        picking argmax of the next-step logits. Use the KV cache: feed the FULL
        prompt on the first call, then ONE token per subsequent call.
        Stop generating for a sequence (and pad with eos_token if provided) once
        eos_token is emitted, OR after max_new_tokens new tokens.
        Returns a LongTensor of shape (B, T_prompt + N_generated), where each row
        contains the prompt followed by generated tokens (truncated or padded with
        eos_token after the row's EOS).

For simplicity assume all sequences in the batch may continue independently; once
ALL of them have produced EOS, return early.
"""

import torch


def greedy_generate(step_fn, prompt_ids, max_new_tokens, eos_token=None):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
