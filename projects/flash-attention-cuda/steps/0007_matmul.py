"""
Step 0007: matmul

Part 2 — Matrix Operations
C = A @ B for A (M,K), B (K,N) -> (M,N).

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

You edit the CUDA inside the matmul string below. The host function must stay
named `matmul` and keep its signature: torch::Tensor matmul(torch::Tensor,torch::Tensor)
"""
matmul = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<M && c<N) C[r*N+c] = sum_k A[r*K+k]*B[k*N+c]
}
torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0), K=A.size(1), N=B.size(1); auto C = torch::empty({M,N}, A.options());
    dim3 t(16,16), bl((N+15)/16,(M+15)/16);
    // TODO: launch matmul_kernel<<<bl,t>>>(...)
    return C;
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
