"""Build a Mini vLLM Inference Engine — end-to-end demo.

    python projects.py mini-vllm-inference-engine --scaffold

Serves a tiny attention LM three ways and shows the engine invariants vLLM and
SGLang are built on: PagedAttention (block-paged KV cache) and continuous
batching produce BYTE-IDENTICAL output to plain sequential decoding, a radix
prefix cache (RadixAttention) reuses shared prefixes, and sampling works.
Pure NumPy.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import *  # noqa: F401,F403


def main():
    model = make_model(vocab=16, d=8, seed=0)
    scale = 1.0 / np.sqrt(model["d"])
    block_size, num_blocks = 4, 64

    prompts = {
        "a": [1, 2, 3],
        "b": [1, 2, 7, 4],     # shares the "1 2" prefix with a
        "c": [5, 9],
    }
    max_new = 6

    # (1) Ground-truth sequential decode (contiguous KV cache).
    seq = {rid: greedy_generate(model, p, max_new, scale) for rid, p in prompts.items()}
    print("sequential decode:")
    for rid, g in seq.items():
        print(f"  req {rid}: prompt {prompts[rid]} -> {g}")

    # (2) PagedAttention: same output, KV stored in fixed-size blocks.
    paged_ok = all(
        paged_generate(model, p, max_new, scale, num_blocks, block_size) == seq[rid]
        for rid, p in prompts.items()
    )
    print(f"PagedAttention (block_size={block_size}) == sequential: {paged_ok}")

    # (3) Continuous batching: all requests advance together, one token/step.
    batched = run_batched(model, list(prompts.items()), num_blocks, block_size, scale, max_new)
    batch_ok = all(batched[rid] == seq[rid] for rid in prompts)
    print(f"continuous batching == sequential: {batch_ok}  (served {len(batched)} reqs together)")

    # (4) RadixAttention prefix cache: reuse the longest cached prefix.
    root = {"children": {}}
    radix_insert(root, prompts["a"])
    shared = radix_longest_prefix(root, prompts["b"])
    print(f"radix prefix cache: req b reuses {shared} cached tokens of {prompts['b']} "
          f"(prompt+gen of a already computed)")

    # (5) Sampling: greedy vs temperature.
    logits = model["Wout"][0] * 0 + np.arange(model["vocab"], dtype=float)
    rng = np.random.default_rng(0)
    g = greedy_sample(logits)
    s_lowT = sample_token(logits, temperature=0.01, top_k=0, top_p=1.0, rng=rng)
    s_topk = sample_token(logits, temperature=1.0, top_k=1, top_p=1.0, rng=rng)
    print(f"sampling: greedy={g}  temp~0={s_lowT}  top_k=1={s_topk}  (all should equal argmax={g})")

    ok = paged_ok and batch_ok and shared == 2 and g == s_lowT == s_topk
    print("ok" if ok else "warning")


if __name__ == "__main__":
    main()
