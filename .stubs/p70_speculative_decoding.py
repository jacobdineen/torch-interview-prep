"""
Problem 70: Speculative Decoding (Greedy Variant)

Speculative decoding speeds up LLM inference by using a small "draft" model to
propose K next tokens, then verifying them in parallel with the target model.
Each draft token is accepted iff the target model's argmax at that position
matches the draft. We accept the prefix of matching tokens and append one bonus
token from the target's distribution at the first mismatch position.

Implement the GREEDY (argmax) verification variant:

  - speculative_decode(target_step, draft_step, prompt_ids, max_new_tokens, K=4)
        target_step(input_ids: LongTensor (B, T))    -> logits (B, T, V)
        draft_step(input_ids:  LongTensor (B, T))    -> logits (B, T, V)

        Algorithm (one "speculative round"):
          1. Use draft_step on current sequence to extend by K tokens (greedy).
          2. Run target_step ONCE on the extended sequence to get logits at every
             new position (and one extra position past the last draft token).
          3. For each draft token at position i in [0, K): accept iff
             target_argmax_at_position_i == draft_token_i.
          4. On the first MISMATCH (or after all K accepted), append the target's
             argmax at the first non-matched position (i.e., one bonus token).
          5. Repeat until max_new_tokens generated.

        prompt_ids: (1, T0). Output: (1, T0 + N_generated).

The key efficiency win is step (2): one parallel call of the target model covers
K positions. We accept >=1 and at most K+1 new tokens per round.
"""

import torch


def speculative_decode(target_step, draft_step, prompt_ids, max_new_tokens, K=4):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
