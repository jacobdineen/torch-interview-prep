"""
Step 0026: flash_attention_causal_kernel

Part 7 — Causal Flash Attention
Causal flash attention: query r attends only to keys c <= r.

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

You edit the CUDA inside the flash_attention_causal_kernel string below. The host function must stay
named `flash_attention_causal_kernel` and keep its signature: torch::Tensor flash_attention_causal_kernel(torch::Tensor,torch::Tensor,torch::Tensor,double)
"""
flash_attention_causal_kernel = r'''#include <torch/extension.h>
#include <cuda_runtime.h>

// Causal flash attention (self-attention, N==M): query r attends only to keys c <= r.
__global__ void flash_attention_causal_kernel_k(const float* Q, const float* K, const float* V,
                                                float* O, float sc, int N, int d) {
    int r = blockIdx.x; if (r >= N) return; if (threadIdx.x != 0) return;
    float m=-1e30f,l=0.f,acc[128]; for(int j=0;j<d;j++)acc[j]=0.f;
    // TODO: same online softmax as flash, but loop keys c from 0 to r (inclusive)
    for(int j=0;j<d;j++) O[r*d+j]=acc[j]/l;
}
torch::Tensor flash_attention_causal_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double sc) {
    int N=Q.size(0), d=Q.size(1);
    auto O = torch::empty({N,d}, Q.options());
    flash_attention_causal_kernel_k<<<N,1>>>(Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),
                                             O.data_ptr<float>(),(float)sc,N,d);
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
