"""Generate CUDA-string step stubs, project.json, and compiled tests for the
flash-attention-cuda project.  python _build/gen.py
"""
import ast
import importlib.util
import json
import os
import py_compile
import sys

BUILD = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(BUILD)
REF = os.path.join(PROJECT, "tests", "_ref", "reference.py")
TESTS = os.path.join(PROJECT, "tests", "_ref", "tests.py")
COMPILED = os.path.join(PROJECT, "tests", "_compiled")
STEPS = os.path.join(PROJECT, "steps")
sys.path.insert(0, BUILD)
import spec  # noqa: E402

DOCS = {
 "vector_add": "Elementwise add two 1-D arrays: out[i] = a[i] + b[i].",
 "scale_array": "Multiply every element of a 1-D array by scalar s.",
 "elementwise_exp": "Apply expf elementwise: out[i] = exp(a[i]).",
 "row_max": "Reduce each row of a (R,C) matrix to its max -> (R,).",
 "row_sum": "Reduce each row of a (R,C) matrix to its sum -> (R,).",
 "dot_product": "Dot product of two 1-D vectors -> scalar tensor.",
 "matmul": "C = A @ B for A (M,K), B (K,N) -> (M,N).",
 "transpose": "Transpose a (M,N) matrix -> (N,M).",
 "qk_scores": "Scaled scores S = scale * Q @ K^T; Q (N,d), K (M,d) -> (N,M).",
 "softmax_rows": "Numerically-stable row-wise softmax of a (R,C) matrix.",
 "pv_matmul": "O = P @ V for P (N,M), V (M,d) -> (N,d).",
 "naive_attention": "Full attention softmax(scale * Q@K^T) @ V -> (N,d).",
 "online_max": "Elementwise running max: out[i] = max(m[i], x[i]).",
 "correction_factor": "Rescale factor exp(m_old - m_new), elementwise.",
 "update_running_sum": "Online denominator: out = alpha*l + s, elementwise.",
 "rescale_output": "Scale each row r of O (R,d) by alpha[r].",
 "load_tile": "Stage an (R,C) tile from global memory (copy it through).",
 "tile_scores": "Tile-level scaled scores: scale * Q @ K^T.",
 "tile_rowmax": "Row max of a score tile -> (R,).",
 "tile_exp": "exp(S[r,c] - m[r]) over a score tile.",
 "tile_rowsum": "Row sum of an exp'd tile -> (R,).",
 "accumulate_pv": "Tile PV product P @ V -> (R,d).",
 "flash_attention_kernel": "The fused online-softmax attention KERNEL (host launcher is given).",
 "flash_attention_launcher": "Host launcher for the flash attention kernel (kernel is given).",
 "causal_mask": "Set S[r,c] = -1e30f where c > r (mask future positions).",
 "flash_attention_causal_kernel": "Causal flash attention: query r attends only to keys c <= r.",
}

HDR = "#include <torch/extension.h>\n#include <cuda_runtime.h>\n"
_ELT2 = '''
__global__ void {n}_kernel(const float* a, const float* b, float* out, int n) {{
    // TODO: i = blockIdx.x*blockDim.x + threadIdx.x; if (i<n) out[i] = ...;
}}
torch::Tensor {n}(torch::Tensor a, torch::Tensor b) {{
    auto out = torch::empty_like(a); int n = a.numel();
    // TODO: launch {n}_kernel with (n+255)/256 blocks of 256 threads
    return out;
}}'''
SKEL = {
 "vector_add": HDR + _ELT2.format(n="vector_add"),
 "scale_array": HDR + '''
__global__ void scale_array_kernel(const float* a, float s, float* out, int n) {
    // TODO: out[i] = a[i] * s
}
torch::Tensor scale_array(torch::Tensor a, double s) {
    auto out = torch::empty_like(a); int n = a.numel();
    // TODO: launch with (float)s
    return out;
}''',
 "elementwise_exp": HDR + '''
__global__ void elementwise_exp_kernel(const float* a, float* out, int n) {
    // TODO: out[i] = expf(a[i])
}
torch::Tensor elementwise_exp(torch::Tensor a) {
    auto out = torch::empty_like(a); int n = a.numel();
    // TODO: launch
    return out;
}''',
 "row_max": HDR + '''
// One block per row; reduce the row to a single max in shared memory.
__global__ void row_max_kernel(const float* a, float* out, int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    // TODO: each thread maxes a stride of the row, then reduce in __shared__; thread 0 writes out[r]
}
torch::Tensor row_max(torch::Tensor a) {
    int R = a.size(0), C = a.size(1); auto out = torch::empty({R}, a.options());
    // TODO: launch <<<R, 256>>>
    return out;
}''',
 "row_sum": HDR + '''
__global__ void row_sum_kernel(const float* a, float* out, int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    // TODO: reduce the row to a sum (shared-memory reduction); thread 0 writes out[r]
}
torch::Tensor row_sum(torch::Tensor a) {
    int R = a.size(0), C = a.size(1); auto out = torch::empty({R}, a.options());
    // TODO: launch <<<R, 256>>>
    return out;
}''',
 "dot_product": HDR + '''
__global__ void dot_product_kernel(const float* a, const float* b, float* out, int n) {
    // TODO: per-block partial sum of a[i]*b[i] in shared memory, then atomicAdd to out
}
torch::Tensor dot_product(torch::Tensor a, torch::Tensor b) {
    auto out = torch::zeros({}, a.options()); int n = a.numel();
    // TODO: launch
    return out;
}''',
 "matmul": HDR + '''
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<M && c<N) C[r*N+c] = sum_k A[r*K+k]*B[k*N+c]
}
torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0), K=A.size(1), N=B.size(1); auto C = torch::empty({M,N}, A.options());
    dim3 t(16,16), bl((N+15)/16,(M+15)/16);
    // TODO: launch matmul_kernel<<<bl,t>>>(...)
    return C;
}''',
 "transpose": HDR + '''
__global__ void transpose_kernel(const float* A, float* O, int M, int N) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<M && c<N) O[c*M + r] = A[r*N + c]
}
torch::Tensor transpose(torch::Tensor A) {
    int M=A.size(0), N=A.size(1); auto O = torch::empty({N,M}, A.options());
    dim3 t(16,16), bl((N+15)/16,(M+15)/16);
    // TODO: launch
    return O;
}''',
 "qk_scores": HDR + '''
__global__ void qk_scores_kernel(const float* Q, const float* K, float* S, float sc, int N, int M, int d) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<N && c<M) S[r*M+c] = sc * sum_k Q[r*d+k]*K[c*d+k]
}
torch::Tensor qk_scores(torch::Tensor Q, torch::Tensor K, double sc) {
    int N=Q.size(0), d=Q.size(1), M=K.size(0); auto S = torch::empty({N,M}, Q.options());
    dim3 t(16,16), bl((M+15)/16,(N+15)/16);
    // TODO: launch with (float)sc
    return S;
}''',
 "softmax_rows": HDR + '''
// Stable softmax per row: subtract row max, exp, divide by the row sum.
__global__ void softmax_rows_kernel(const float* S, float* O, int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    // TODO: row max -> exp(S-max) into O + row sum -> divide O by sum
}
torch::Tensor softmax_rows(torch::Tensor S) {
    int R=S.size(0), C=S.size(1); auto O = torch::empty_like(S);
    // TODO: launch <<<R,256>>>
    return O;
}''',
 "pv_matmul": HDR + '''
__global__ void pv_matmul_kernel(const float* P, const float* V, float* O, int N, int M, int d) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<N && c<d) O[r*d+c] = sum_k P[r*M+k]*V[k*d+c]
}
torch::Tensor pv_matmul(torch::Tensor P, torch::Tensor V) {
    int N=P.size(0), M=P.size(1), d=V.size(1); auto O = torch::empty({N,d}, P.options());
    dim3 t(16,16), bl((d+15)/16,(N+15)/16);
    // TODO: launch
    return O;
}''',
 "naive_attention": HDR + '''
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
}''',
 "online_max": HDR + _ELT2.format(n="online_max").replace("out[i] = ...", "out[i] = fmaxf(m[i], x[i])").replace("(const float* a, const float* b,", "(const float* m, const float* x,").replace("torch::Tensor online_max(torch::Tensor a, torch::Tensor b)", "torch::Tensor online_max(torch::Tensor m, torch::Tensor x)"),
 "correction_factor": HDR + '''
__global__ void correction_factor_kernel(const float* m_old, const float* m_new, float* out, int n) {
    // TODO: out[i] = expf(m_old[i] - m_new[i])
}
torch::Tensor correction_factor(torch::Tensor m_old, torch::Tensor m_new) {
    auto out = torch::empty_like(m_old); int n = m_old.numel();
    // TODO: launch
    return out;
}''',
 "update_running_sum": HDR + '''
__global__ void update_running_sum_kernel(const float* l, const float* alpha, const float* s, float* out, int n) {
    // TODO: out[i] = alpha[i]*l[i] + s[i]
}
torch::Tensor update_running_sum(torch::Tensor l, torch::Tensor alpha, torch::Tensor s) {
    auto out = torch::empty_like(l); int n = l.numel();
    // TODO: launch
    return out;
}''',
 "rescale_output": HDR + '''
__global__ void rescale_output_kernel(const float* O, const float* alpha, float* out, int R, int d) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<R && c<d) out[r*d+c] = alpha[r] * O[r*d+c]
}
torch::Tensor rescale_output(torch::Tensor O, torch::Tensor alpha) {
    int R=O.size(0), d=O.size(1); auto out = torch::empty_like(O);
    dim3 t(16,16), bl((d+15)/16,(R+15)/16);
    // TODO: launch
    return out;
}''',
 "load_tile": HDR + '''
// Stage a tile from global memory. (A real flash kernel keeps it in __shared__;
// here just copy it through so the mechanics are graded.)
__global__ void load_tile_kernel(const float* X, float* O, int R, int C) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<R && c<C) O[r*C+c] = X[r*C+c]
}
torch::Tensor load_tile(torch::Tensor X) {
    int R=X.size(0), C=X.size(1); auto O = torch::empty_like(X);
    dim3 t(16,16), bl((C+15)/16,(R+15)/16);
    // TODO: launch
    return O;
}''',
 "tile_scores": HDR + '''
__global__ void tile_scores_kernel(const float* Q, const float* K, float* S, float sc, int N, int M, int d) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<N && c<M) S[r*M+c] = sc * sum_k Q[r*d+k]*K[c*d+k]
}
torch::Tensor tile_scores(torch::Tensor Q, torch::Tensor K, double sc) {
    int N=Q.size(0), d=Q.size(1), M=K.size(0); auto S = torch::empty({N,M}, Q.options());
    dim3 t(16,16), bl((M+15)/16,(N+15)/16);
    // TODO: launch
    return S;
}''',
 "tile_rowmax": HDR + '''
__global__ void tile_rowmax_kernel(const float* a, float* out, int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    // TODO: shared-memory row-max reduction; thread 0 writes out[r]
}
torch::Tensor tile_rowmax(torch::Tensor a) {
    int R=a.size(0), C=a.size(1); auto out = torch::empty({R}, a.options());
    // TODO: launch <<<R,256>>>
    return out;
}''',
 "tile_exp": HDR + '''
__global__ void tile_exp_kernel(const float* S, const float* m, float* O, int R, int C) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<R && c<C) O[r*C+c] = expf(S[r*C+c] - m[r])
}
torch::Tensor tile_exp(torch::Tensor S, torch::Tensor m) {
    int R=S.size(0), C=S.size(1); auto O = torch::empty_like(S);
    dim3 t(16,16), bl((C+15)/16,(R+15)/16);
    // TODO: launch
    return O;
}''',
 "tile_rowsum": HDR + '''
__global__ void tile_rowsum_kernel(const float* a, float* out, int R, int C) {
    int r = blockIdx.x; if (r >= R) return;
    // TODO: shared-memory row-sum reduction; thread 0 writes out[r]
}
torch::Tensor tile_rowsum(torch::Tensor a) {
    int R=a.size(0), C=a.size(1); auto out = torch::empty({R}, a.options());
    // TODO: launch <<<R,256>>>
    return out;
}''',
 "accumulate_pv": HDR + '''
__global__ void accumulate_pv_kernel(const float* P, const float* V, float* O, int N, int M, int d) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<N && c<d) O[r*d+c] = sum_k P[r*M+k]*V[k*d+c]
}
torch::Tensor accumulate_pv(torch::Tensor P, torch::Tensor V) {
    int N=P.size(0), M=P.size(1), d=V.size(1); auto O = torch::empty({N,d}, P.options());
    dim3 t(16,16), bl((d+15)/16,(N+15)/16);
    // TODO: launch
    return O;
}''',
 "flash_attention_kernel": HDR + '''
// YOU WRITE THE KERNEL. One block per query row; iterate keys keeping the online
// softmax running max m, running denom l, and running output acc[d].
__global__ void flash_attention_kernel_k(const float* Q, const float* K, const float* V,
                                         float* O, float sc, int N, int M, int d) {
    int r = blockIdx.x; if (r >= N) return; if (threadIdx.x != 0) return;
    float m = -1e30f, l = 0.f, acc[128];
    for (int j=0;j<d;j++) acc[j]=0.f;
    // TODO: for each key c: s = sc*dot(Q[r],K[c]); m_new=max(m,s); corr=exp(m-m_new);
    //       p=exp(s-m_new); l = corr*l + p; acc[j] = corr*acc[j] + p*V[c,j]; m=m_new;
    // then O[r,j] = acc[j]/l
}
torch::Tensor flash_attention_kernel(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double sc) {
    int N=Q.size(0), d=Q.size(1), M=K.size(0); auto O = torch::empty({N,d}, Q.options());
    flash_attention_kernel_k<<<N,1>>>(Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),
                                      O.data_ptr<float>(),(float)sc,N,M,d);
    return O;
}''',
 "flash_attention_launcher": HDR + '''
// The kernel is given; YOU WRITE THE LAUNCHER (allocate O, pick the launch config,
// call the kernel, return O).
__global__ void flash_attention_launcher_k(const float* Q, const float* K, const float* V,
                                           float* O, float sc, int N, int M, int d) {
    int r = blockIdx.x; if (r >= N) return; if (threadIdx.x != 0) return;
    float m=-1e30f,l=0.f,acc[128]; for(int j=0;j<d;j++)acc[j]=0.f;
    for(int c=0;c<M;c++){float s=0;for(int k=0;k<d;k++)s+=Q[r*d+k]*K[c*d+k];s*=sc;
      float mn=fmaxf(m,s),corr=expf(m-mn),p=expf(s-mn);l=corr*l+p;
      for(int j=0;j<d;j++)acc[j]=corr*acc[j]+p*V[c*d+j];m=mn;}
    for(int j=0;j<d;j++)O[r*d+j]=acc[j]/l;
}
torch::Tensor flash_attention_launcher(torch::Tensor Q, torch::Tensor K, torch::Tensor V, double sc) {
    int N=Q.size(0), d=Q.size(1), M=K.size(0);
    auto O = torch::empty({N,d}, Q.options());
    // TODO: launch flash_attention_launcher_k<<<N,1>>>(... , (float)sc, N, M, d)
    return O;
}''',
 "causal_mask": HDR + '''
__global__ void causal_mask_kernel(const float* S, float* O, int N, int M) {
    int r = blockIdx.y*blockDim.y + threadIdx.y, c = blockIdx.x*blockDim.x + threadIdx.x;
    // TODO: if (r<N && c<M) O[r*M+c] = (c > r) ? -1e30f : S[r*M+c]
}
torch::Tensor causal_mask(torch::Tensor S) {
    int N=S.size(0), M=S.size(1); auto O = torch::empty_like(S);
    dim3 t(16,16), bl((M+15)/16,(N+15)/16);
    // TODO: launch
    return O;
}''',
 "flash_attention_causal_kernel": HDR + '''
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
}''',
}

STUB = '''"""
Step {id}: {name}

Part {pnum} — {ptitle}
{doc}
{primer}
You edit the CUDA inside the {name} string below. The host function must stay
named `{name}` and keep its signature: {sig}
"""
{name} = r\'\'\'{skel}
\'\'\'


if __name__ == "__main__":
    import os
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step
    raise SystemExit(run_step(os.path.abspath(__file__)))
'''


def _sig(name):
    ref = importlib.util.spec_from_file_location("ref", REF)
    return None  # filled below


def main():
    refmod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("ref_src", REF))
    importlib.util.spec_from_file_location("ref_src", REF).loader.exec_module(refmod)
    os.makedirs(STEPS, exist_ok=True); os.makedirs(COMPILED, exist_ok=True)
    manifest_steps = []
    written = 0
    for i, (name, part) in enumerate(spec.STEPS, start=1):
        sid = spec.step_id(i)
        sig = refmod.STEP_SPECS[name]["decl"].rstrip(";")
        doc = DOCS.get(name, "")
        manifest_steps.append({"id": sid, "name": name, "part": part, "points": 5,
                               "signature": sig, "doc": doc})
        path = os.path.join(STEPS, f"{sid}_{name}.py")
        if os.path.exists(path):
            continue
        with open(path, "w") as f:
            f.write(STUB.format(id=sid, name=name, pnum=part + 1, ptitle=spec.PARTS[part][0],
                                doc=doc, primer=spec.PRIMER, sig=sig, skel=SKEL[name]))
        written += 1
    manifest = {"name": os.path.basename(PROJECT), "title": getattr(spec, "TITLE", "flash-attention-cuda"),
                "primer": spec.PRIMER,
                "parts": [{"title": t, "description": d} for t, d in spec.PARTS],
                "steps": manifest_steps}
    with open(os.path.join(PROJECT, "project.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    py_compile.compile(REF, cfile=os.path.join(COMPILED, "reference.pyc"), doraise=True)
    py_compile.compile(TESTS, cfile=os.path.join(COMPILED, "tests.pyc"), doraise=True)
    print(f"manifest: {len(manifest_steps)} steps | stubs written: {written}")
    print("compiled: reference.pyc, tests.pyc")


if __name__ == "__main__":
    main()
