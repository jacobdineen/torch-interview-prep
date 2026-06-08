"""
Step 0025: causal_mask

Part 7 — Causal Flash Attention
Set S[r,c] = -1e30f where c > r (mask future positions).

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

You edit the CUDA inside the causal_mask string below. The host function must stay
named `causal_mask` and keep its signature: torch::Tensor causal_mask(torch::Tensor)
"""
causal_mask = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void causal_mask_kernel(const float* S, float* O, int N, int M) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<N && c<M) O[r*M+c] = (c > r) ? -1e30f : S[r*M+c]
}
torch::Tensor causal_mask(torch::Tensor S) {
    int N=S.size(0), M=S.size(1); auto O = torch::empty_like(S);
    dim3 t(16,16), bl((M+15)/16,(N+15)/16);
    // TODO: launch
    return O;
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
