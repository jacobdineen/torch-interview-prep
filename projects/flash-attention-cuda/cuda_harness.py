"""Self-contained CUDA JIT-compile harness for the flash-attention-cuda project.

Grading a CUDA step = compile the learner's kernel source and run it on the GPU,
then compare to a torch oracle. This sets up the toolchain env itself (CUDA 12.3
needs g++-12 as the host compiler; ninja lives in the venv) so a bare
`python steps/NNNN_*.py` just works — no manual `export` needed.
"""
import hashlib
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _setup_env():
    cuda = "/usr/local/cuda"
    os.environ.setdefault("CUDA_HOME", cuda)
    # CUDA 12.3's host_config rejects gcc>12; g++-12 is the supported host compiler.
    for var, val in (("CC", "gcc-12"), ("CXX", "g++-12")):
        os.environ.setdefault(var, val)
    # ninja lives next to the running interpreter (works from any checkout,
    # e.g. a git worktree without its own .venv) or in the repo's .venv.
    import sys
    venv_bin = os.path.join(_REPO, ".venv", "bin")
    extra = os.pathsep.join([os.path.dirname(sys.executable), venv_bin,
                             os.path.join(cuda, "bin")])
    if extra not in os.environ.get("PATH", ""):
        os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")


def have_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_cache = {}


def compile_cuda(cuda_source, functions, cpp_decls=None, tag=""):
    """JIT-compile a CUDA source exposing `functions` (host wrappers). cpp_decls is
    the C++ signature(s) the pybind layer needs; defaults to extern decls derived
    from the names. Returns the loaded module. Cached by source hash."""
    _setup_env()
    from torch.utils.cpp_extension import load_inline
    if cpp_decls is None:
        cpp_decls = "\n".join(f"torch::Tensor {f}(torch::Tensor);" for f in functions)
    key = hashlib.md5((tag + cuda_source + cpp_decls + ",".join(functions)).encode()).hexdigest()[:16]
    if key in _cache:
        return _cache[key]
    name = f"fa_{tag or 'mod'}_{key}"
    mod = load_inline(name=name, cpp_sources=cpp_decls, cuda_sources=cuda_source,
                      functions=list(functions), with_cuda=True, verbose=False,
                      extra_cuda_cflags=["-O2", "-ccbin", "g++-12",
                                         "--expt-relaxed-constexpr"])
    _cache[key] = mod
    return mod
