# Tensor Ops with einops

Learn [einops](https://einops.rocks) by re-deriving the everyday tensor moves of
deep learning — transposing, flattening, splitting into attention heads, pooling,
broadcasting, batched matmuls — as single declarative calls instead of brittle
chains of `transpose`/`reshape`/`sum`.

Every step is one function in `steps/NNNN_<fn>.py`, solved with **one** einops
call and graded in isolation against a hidden reference. The hidden tests check
your output against an independent plain-NumPy computation, so passing means the
operation is genuinely correct, not just self-consistent. All arrays are NumPy —
einops is framework-agnostic, so the same patterns work unchanged on PyTorch
tensors.

## The five tools

| Part | einops call | What you practise |
|------|-------------|-------------------|
| 1. Rearrange | `rearrange` | transpose, flatten, merge/split axes, NHWC↔NCHW |
| 2. Reduce | `reduce` | global / patch mean-, max-, sum-pooling |
| 3. Repeat | `repeat` | broadcast a new axis, tile, nearest-neighbour upsample |
| 4. Einsum | `einsum` | batched matmul, attention scores, weighted sums, trace |
| 5. Pack | `pack` | flatten/concatenate tensors of ragged rank |

The stubs already `import numpy as np` and pull in `rearrange, reduce, repeat,
einsum, pack, unpack` — the docstring tells you the shape transform to produce;
you supply the pattern string.

## Working through it

```bash
uv run python projects.py einops               # parts + steps, [x]/[ ] solved
uv run python projects.py einops --next        # jump to the next unsolved step
uv run python projects.py einops 0004 --explain  # task + a worked input/output example
uv run python projects.py einops --scaffold    # end-to-end demo (once solved)
```

Grade a single step by running its file directly:

```bash
uv run python projects/einops/steps/0001_transpose_2d.py
```
