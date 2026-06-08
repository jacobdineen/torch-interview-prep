"""Verify every reference kernel compiles + matches its torch oracle (run on GPU).
Skips cleanly with exit 0 when there is no GPU, so verify_all/CI pass on CPU hosts.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, "tests", "_ref"))
sys.path.insert(0, PROJECT)
import reference as R  # noqa: E402


def main():
    if not R.have_gpu():
        print("flash-attention-cuda: no CUDA GPU detected — skipping (refs need a GPU).")
        return 0
    import torch
    from cuda_harness import compile_cuda
    names = list(R.STEP_SPECS)
    combined = "\n".join(getattr(R, n) for n in names)
    decls = "\n".join(R.STEP_SPECS[n]["decl"] for n in names)
    mod = compile_cuda(combined, names, cpp_decls=decls, tag="verify")
    npass, fails = 0, []
    for n in names:
        sp = R.STEP_SPECS[n]; inp = sp["inp"](); out = getattr(mod, n)(*inp); exp = sp["ora"](*inp)
        atol = sp.get("atol", 1e-4); rtol = sp.get("rtol", 1e-3)
        if tuple(out.shape) == tuple(exp.shape) and torch.allclose(out, exp, atol=atol, rtol=rtol):
            npass += 1
        else:
            fails.append(n)
    print(f"{npass} pass, {len(fails)} fail, 0 not-yet-built" + (f"  FAILED: {fails}" if fails else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
