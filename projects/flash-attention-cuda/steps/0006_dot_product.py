"""
Step 0006: dot_product

Part 2 — Matrix Operations
Dot product of two 1-D vectors -> scalar tensor.

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

You edit the CUDA inside the dot_product string below. The host function must stay
named `dot_product` and keep its signature: torch::Tensor dot_product(torch::Tensor,torch::Tensor)
"""
dot_product = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void dot_product_kernel(const float* a, const float* b, float* out, int n) {
    // TODO: per-block partial sum of a[i]*b[i] in shared memory, then atomicAdd to out
}
torch::Tensor dot_product(torch::Tensor a, torch::Tensor b) {
    auto out = torch::zeros({}, a.options()); int n = a.numel();
    // TODO: launch
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
