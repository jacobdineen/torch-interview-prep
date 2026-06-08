"""
Step 0012: naive_attention

Part 3 — Naive Attention Baseline
Full attention softmax(scale * Q@K^T) @ V -> (N,d).

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

You edit the CUDA inside the naive_attention string below. The host function must stay
named `naive_attention` and keep its signature: torch::Tensor naive_attention(torch::Tensor,torch::Tensor,torch::Tensor,double)
"""
naive_attention = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

// One block per query row: compute its M scores (shared), stable-softmax them, weight V.
__global__ void naive_attention_kernel(const float* Q, const float* K, const float* V,
                                       float* O, float sc, int N, int M, int d) {
    int r = blockIdx.x; if (r >= N) return;
    extern __shared__ float scr[];   // M scores for this row
    // TODO: scores -> rowmax -> exp -> rowsum -> O[r,:] = (sum_c p_c * V[c,:]) / Z
}
torch::Tensor naive_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double sc) {
    int N=Q.size(0), d=Q.size(1), M=K.size(0); auto O = torch::empty({N,d}, Q.options());
    // TODO: launch <<<N, 256, M*sizeof(float)>>>
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
