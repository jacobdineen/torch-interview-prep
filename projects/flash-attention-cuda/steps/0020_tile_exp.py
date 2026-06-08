"""
Step 0020: tile_exp

Part 5 — Tiled Attention Building Blocks
exp(S[r,c] - m[r]) over a score tile.

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

You edit the CUDA inside the tile_exp string below. The host function must stay
named `tile_exp` and keep its signature: torch::Tensor tile_exp(torch::Tensor,torch::Tensor)
"""
tile_exp = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tile_exp_kernel(const float* S, const float* m, float* O, int R, int C) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<R && c<C) O[r*C+c] = expf(S[r*C+c] - m[r])
}
torch::Tensor tile_exp(torch::Tensor S, torch::Tensor m) {
    int R=S.size(0), C=S.size(1); auto O = torch::empty_like(S);
    dim3 t(16,16), bl((C+15)/16,(R+15)/16);
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
