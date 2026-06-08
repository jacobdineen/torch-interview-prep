"""
Step 0010: softmax_rows

Part 3 — Naive Attention Baseline
Numerically-stable row-wise softmax of a (R,C) matrix.

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

You edit the CUDA inside the softmax_rows string below. The host function must stay
named `softmax_rows` and keep its signature: torch::Tensor softmax_rows(torch::Tensor)
"""
softmax_rows = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

// Stable softmax per row: subtract row max, exp, divide by the row sum.
__global__ void softmax_rows_kernel(const float* S, float* O, int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    // TODO: row max -> exp(S-max) into O + row sum -> divide O by sum
}
torch::Tensor softmax_rows(torch::Tensor S) {
    int R=S.size(0), C=S.size(1); auto O = torch::empty_like(S);
    // TODO: launch <<<R,256>>>
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
