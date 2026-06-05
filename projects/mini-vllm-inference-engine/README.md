# Build a Mini vLLM Inference Engine

Build the *serving engine* behind vLLM and SGLang — not the model, the machinery
that makes LLM inference fast. You implement **PagedAttention** (a block-paged KV
cache with a free-block pool and per-sequence block tables), **continuous
batching** (admit, prefill, and advance many requests of different lengths
together), and **RadixAttention** (a token-trie prefix cache for KV reuse), plus
greedy / temperature / top-k / top-p sampling. **20 steps, 5 parts.**

| Part | Focus |
|------|-------|
| 1 | KV-cache autoregressive decode over a tiny attention LM (the workload) |
| 2 | PagedAttention paging: block manager, allocate/free, logical→physical slots |
| 3 | Paged compute: write/gather KV through the block table, paged decode |
| 4 | Continuous batching: admit, prefill, batched decode step, the engine loop |
| 5 | RadixAttention prefix cache + sampling |

The correctness bar is an *invariant*: paged decode and continuous batching must
produce output **identical** to plain sequential decoding — the same property the
real systems rely on. A tiny single-head attention LM is provided; you build
everything that schedules and serves it.

```bash
uv run python projects.py mini-vllm-inference-engine            # parts + steps
uv run python projects.py mini-vllm-inference-engine --scaffold # paged/batched/radix/sampling demo
```
