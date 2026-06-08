"""Build Flash Attention in CUDA from Scratch — end-to-end demo.

    python projects.py flash-attention-cuda --scaffold

Compiles the reference naive + flash + causal kernels, runs them on a toy
attention problem, and confirms they match PyTorch to GPU precision. Needs a CUDA GPU.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tests", "_ref"))
sys.path.insert(0, HERE)
import reference as R  # noqa: E402


def main():
    if not R.have_gpu():
        print("No CUDA GPU detected — the CUDA kernels need a GPU to run.")
        return
    import torch
    from cuda_harness import compile_cuda
    names = ["naive_attention", "flash_attention_kernel", "flash_attention_causal_kernel"]
    combined = "\n".join(getattr(R, n) for n in names)
    decls = "\n".join(R.STEP_SPECS[n]["decl"] for n in names)
    print("compiling reference kernels (first time ~60s)…")
    mod = compile_cuda(combined, names, cpp_decls=decls, tag="scaffold")

    N, M, d = 128, 128, 64
    scale = 1.0 / (d ** 0.5)
    Q = torch.randn(N, d, device="cuda"); K = torch.randn(M, d, device="cuda"); V = torch.randn(M, d, device="cuda")
    ref = torch.softmax(scale * (Q @ K.t()), dim=1) @ V
    naive = mod.naive_attention(Q, K, V, scale)
    flash = mod.flash_attention_kernel(Q, K, V, scale)
    print(f"  naive_attention  vs torch:  max|diff| = {(naive-ref).abs().max():.2e}")
    print(f"  flash_attention  vs torch:  max|diff| = {(flash-ref).abs().max():.2e}  (online softmax, never builds the full {N}x{M} matrix)")

    Qc = torch.randn(N, d, device="cuda"); Kc = torch.randn(N, d, device="cuda"); Vc = torch.randn(N, d, device="cuda")
    mask = torch.triu(torch.ones(N, N, device="cuda", dtype=torch.bool), 1)
    cref = torch.softmax((scale * (Qc @ Kc.t())).masked_fill(mask, float("-inf")), dim=1) @ Vc
    causal = mod.flash_attention_causal_kernel(Qc, Kc, Vc, scale)
    print(f"  causal flash     vs torch:  max|diff| = {(causal-cref).abs().max():.2e}")
    ok = (flash - ref).abs().max() < 2e-3 and (causal - cref).abs().max() < 2e-3
    print("ok" if ok else "warning: drift")


if __name__ == "__main__":
    main()
