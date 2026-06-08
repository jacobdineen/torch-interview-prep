"""
Step 0016: rescale_output

Part 4 — Online Softmax Math
Scale each row r of O (R,d) by alpha[r].

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

You edit the CUDA inside the rescale_output string below. The host function must stay
named `rescale_output` and keep its signature: torch::Tensor rescale_output(torch::Tensor,torch::Tensor)
"""
rescale_output = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void rescale_output_kernel(const float* O, const float* alpha, float* out, int R, int d) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<R && c<d) out[r*d+c] = alpha[r] * O[r*d+c]
}
torch::Tensor rescale_output(torch::Tensor O, torch::Tensor alpha) {
    int R=O.size(0), d=O.size(1); auto out = torch::empty_like(O);
    dim3 t(16,16), bl((d+15)/16,(R+15)/16);
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
