# Build Flash Attention from Scratch

Reimplement the core of **FlashAttention** in pure NumPy. The trick is *online
softmax*: a running max and running denominator let you fold one key/value block
into the result at a time, so you compute exact attention while only ever holding
small blocks — never the full N×M score matrix. **18 steps, 5 parts.**

| Part | Focus |
|------|-------|
| 1 | Standard attention baseline (scores, stable softmax, full attention) |
| 2 | Online softmax: running max, rescaling correction, the fused block update |
| 3 | Tiled flash forward — matches standard attention, saves the log-sum-exp |
| 4 | Causal (autoregressive) flash attention with per-block masking |
| 5 | The backward pass — recompute the softmax from the saved L, not the N×M P |

The forward output matches standard attention to ~1e-10; the backward gradients
match finite differences. The saved per-row log-sum-exp `L` is the O(N) state
that replaces storing the O(N²) probability matrix — exactly the memory win that
makes FlashAttention fast.

```bash
uv run python projects.py flash-attention-from-scratch            # parts + steps
uv run python projects.py flash-attention-from-scratch --scaffold # forward/causal/backward checks
```
