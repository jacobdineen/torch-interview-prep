"""
Step 0011: paged_attention

Part 3 — PagedAttention: Paged Compute
Compute attention of q over the paged KV pool, equivalent to contiguous kv_attention.

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


def paged_attention(q, k_pool, v_pool, block_table, num_tokens, block_size, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
