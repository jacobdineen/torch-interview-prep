# Build Flash Attention in CUDA from Scratch

Implement Flash Attention as **real CUDA kernels**, from elementary GPU primitives up
to a fused, IO-aware online-softmax attention kernel and a causal variant. **26 steps,
7 parts.**

Each step is a Python file holding **one CUDA source string** named for the step. You
edit the CUDA inside the string; running the step JIT-compiles it with `nvcc` and runs
the host function on the GPU, comparing to a PyTorch oracle.

| Part | Focus |
|------|-------|
| 1 | CUDA primitives: vector_add, scale, exp, row max/sum reductions |
| 2 | Matrix ops: dot_product, matmul, transpose |
| 3 | Naive attention: QK^T scores, row softmax, PV, full pipeline |
| 4 | Online softmax math: running max, correction factor, running sum, output rescale |
| 5 | Tiled building blocks: load/score/rowmax/exp/rowsum/accumulate per tile |
| 6 | Fused Flash Attention kernel + host launcher |
| 7 | Causal Flash Attention |

**Requirements:** an NVIDIA GPU, a CUDA toolkit (tested with 12.3), `g++-12` (CUDA 12.x
rejects gcc-13 as the host compiler), and `ninja` (installed by `uv sync`). The harness
sets the toolchain env itself, so a bare `python steps/NNNN_*.py` just works. The first
compile takes ~30-60s; results are cached after.

The host function in each step must keep its exact name (the step name) and signature;
tensors are contiguous float32 on CUDA, indexed as `ptr[r*C + c]`.

```bash
uv run python projects.py flash-attention-cuda            # parts + steps
uv run python projects.py flash-attention-cuda --scaffold # compile refs + check vs torch
```
