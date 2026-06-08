"""
Step 0005: row_sum

Part 1 — CUDA Primitives Warm-up
Reduce each row of a (R,C) matrix to its sum -> (R,).

This is a CUDA project: each step is a Python file holding ONE CUDA source string,
named exactly for the step (e.g. the `vector_add` variable holds the CUDA for the
vector_add step). You edit the CUDA inside that string. At grade time the source
is JIT-compiled with nvcc (CUDA 12.3, g++-12 host compiler, set up for you) and
the host function — which MUST be named exactly like the step and take/return
torch tensors — is run on the GPU and compared to a PyTorch oracle.

Tensors are contiguous float32 on CUDA; index a 2-D (R,C) tensor as ptr[r*C + c].
Compiling takes ~30-60s the first time (cached after). Conventions are in each
step's docstring. Run `uv run python projects.py flash-attention-cuda` for the
full outline.

You edit the CUDA inside the row_sum string below. The host function must stay
named `row_sum` and keep its signature: torch::Tensor row_sum(torch::Tensor)
"""
row_sum = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void row_sum_kernel(const float* a, float* out, int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    // TODO: reduce the row to a sum (shared-memory reduction); thread 0 writes out[r]
}
torch::Tensor row_sum(torch::Tensor a) {
    int R = a.size(0), C = a.size(1); auto out = torch::empty({R}, a.options());
    // TODO: launch <<<R, 256>>>
    return out;
}
'''


if __name__ == "__main__":
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step
    raise SystemExit(run_step(os.path.abspath(__file__)))
