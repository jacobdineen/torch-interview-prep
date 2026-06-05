"""Build spec for the mini-vllm-inference-engine project."""

TITLE = 'Build a Mini vLLM Inference Engine'

PARTS = [
    ("KV-Cache Decode", "Autoregressive decoding over a growing KV cache with a tiny single-head attention LM — the workload the engine serves."),
    ("PagedAttention: Paging", "A block manager that hands out fixed-size KV blocks from a free pool, with per-sequence block tables and the logical->physical slot map (vLLM's core idea)."),
    ("PagedAttention: Paged Compute", "Write and gather KV through the block table, attend over the paged cache, and decode a full sequence — output identical to the contiguous cache."),
    ("Continuous Batching", "Admit requests when blocks are free, prefill them, and advance many sequences of different lengths together one token per step."),
    ("Prefix Cache & Sampling", "A radix (token-trie) prefix cache for KV reuse across requests (SGLang's RadixAttention), plus greedy / temperature / top-k / top-p sampling."),
]

STEPS = [
    ("token_qkv", 0), ("kv_attention", 0), ("decode_step", 0), ("greedy_generate", 0),
    ("new_block_manager", 1), ("allocate", 1), ("free", 1), ("slot_for_token", 1),
    ("write_kv", 2), ("gather_kv", 2), ("paged_attention", 2), ("paged_generate", 2),
    ("can_admit", 3), ("prefill_request", 3), ("batch_decode_step", 3), ("run_batched", 3),
    ("radix_insert", 4), ("radix_longest_prefix", 4), ("greedy_sample", 4), ("sample_token", 4),
]


def step_id(i):
    return f"{i:04d}"
