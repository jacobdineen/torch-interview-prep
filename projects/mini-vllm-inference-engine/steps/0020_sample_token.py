"""
Step 0020: sample_token

Part 5 — Prefix Cache & Sampling
Sample a token id from logits with temperature, top-k, and nucleus (top-p) filtering.

If temperature <= 0, return greedy_sample(logits) (argmax). Otherwise scale logits by 1/temperature then softmax. A top_k of 0 (falsy) disables top-k; otherwise keep only the top_k largest logits (rest -inf). top_p >= 1.0 disables nucleus filtering; otherwise keep the smallest set of highest-probability tokens whose cumulative prob reaches top_p (at least the top token), renormalize, and draw with rng.choice(len(logits), p=probs). Return a python int.

Provided for you (already defined in your namespace at grade time — use, do NOT redefine):

  make_model(vocab=16, d=8, seed=0) -> model: a tiny single-head attention LM, a dict:
      model["vocab"], model["d"]              the sizes
      model["E"]    (vocab, d)                token embeddings
      model["Wq"], model["Wk"], model["Wv"]   (d, d)  query / key / value projections
      model["Wout"] (d, vocab)                maps an attention output to next-token logits

Data structures shared across steps:
  block manager:     {"free": [block ids], "block_size": int, "num_blocks": int}
  running request:   {"id", "block_table", "num_tokens", "cur_logits", "generated", "steps_left"}
  radix (prefix) node: {"children": {token_id: node}}
The KV pools k_pool, v_pool are arrays of shape (num_blocks, block_size, d).
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py mini-vllm-inference-engine` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def sample_token(logits, temperature, top_k, top_p, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
