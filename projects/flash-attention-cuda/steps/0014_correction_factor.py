"""
Step 0014: correction_factor

Part 4 — Online Softmax Math
Rescale factor exp(m_old - m_new), elementwise.

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

You edit the CUDA inside the correction_factor string below. The host function must stay
named `correction_factor` and keep its signature: torch::Tensor correction_factor(torch::Tensor,torch::Tensor)
"""
correction_factor = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void correction_factor_kernel(const float* m_old, const float* m_new, float* out, int n) {
    // TODO: out[i] = expf(m_old[i] - m_new[i])
}
torch::Tensor correction_factor(torch::Tensor m_old, torch::Tensor m_new) {
    auto out = torch::empty_like(m_old); int n = m_old.numel();
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
