"""
Step 0001: vector_add

Part 1 — CUDA Primitives Warm-up
Elementwise add two 1-D arrays: out[i] = a[i] + b[i].

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

You edit the CUDA inside the vector_add string below. The host function must stay
named `vector_add` and keep its signature: torch::Tensor vector_add(torch::Tensor,torch::Tensor)
"""
vector_add = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    // TODO: i = blockIdx.x*blockDim.x + threadIdx.x; if (i<n) out[i] = ...;
}
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a); int n = a.numel();
    // TODO: launch vector_add_kernel with (n+255)/256 blocks of 256 threads
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
