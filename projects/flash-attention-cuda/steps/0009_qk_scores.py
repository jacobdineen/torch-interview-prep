"""
Step 0009: qk_scores

Part 3 — Naive Attention Baseline
Scaled scores S = scale * Q @ K^T; Q (N,d), K (M,d) -> (N,M).

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

You edit the CUDA inside the qk_scores string below. The host function must stay
named `qk_scores` and keep its signature: torch::Tensor qk_scores(torch::Tensor,torch::Tensor,double)
"""
qk_scores = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void qk_scores_kernel(const float* Q, const float* K, float* S, float sc, int N, int M, int d) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<N && c<M) S[r*M+c] = sc * sum_k Q[r*d+k]*K[c*d+k]
}
torch::Tensor qk_scores(torch::Tensor Q, torch::Tensor K, double sc) {
    int N=Q.size(0), d=Q.size(1), M=K.size(0); auto S = torch::empty({N,M}, Q.options());
    dim3 t(16,16), bl((M+15)/16,(N+15)/16);
    // TODO: launch with (float)sc
    return S;
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
