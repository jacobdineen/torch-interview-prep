"""
Step 0002: scale_array

Part 1 — CUDA Primitives Warm-up
Multiply every element of a 1-D array by scalar s.

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

You edit the CUDA inside the scale_array string below. The host function must stay
named `scale_array` and keep its signature: torch::Tensor scale_array(torch::Tensor,double)
"""
scale_array = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scale_array_kernel(const float* a, float s, float* out, int n) {
    // TODO: out[i] = a[i] * s
}
torch::Tensor scale_array(torch::Tensor a, double s) {
    auto out = torch::empty_like(a); int n = a.numel();
    // TODO: launch with (float)s
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
