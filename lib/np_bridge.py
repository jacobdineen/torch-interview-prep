"""Grade a NumPy solution against a problem's existing (torch) test.

Most problems are pure tensor->tensor maps, so we don't need a second NumPy
reference + test: we wrap the learner's NumPy function to present a torch
interface (torch tensors in -> numpy arrays; numpy result -> torch tensor) and
run the *existing* compiled torch test against it. A problem "supports numpy"
exactly when a correct NumPy solution passes this bridge; torch-specific problems
(autograd .backward(), nn.Module, requires_grad) can't be bridged and stay
torch-only.

  from lib.np_bridge import run_numpy
  passed, err = run_numpy("02a", "get_row", numpy_source_string)
"""
import importlib.util
import os
import sys
import types
from importlib.machinery import SourcelessFileLoader

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMPILED = os.path.join(_HERE, "tests", "_compiled")
_PROBLEMS = os.path.join(_HERE, "problems")

# torch is imported lazily so numpy-only environments still import this module.
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _torch_dtype_to_np(dt):
    return {torch.float32: np.float32, torch.float64: np.float64,
            torch.int64: np.int64, torch.int32: np.int32,
            torch.bool: np.bool_, torch.float16: np.float16}.get(dt, None)


def to_np(x):
    """torch tensor -> numpy array; torch.dtype -> numpy dtype; else unchanged."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if torch is not None and isinstance(x, torch.dtype):
        return _torch_dtype_to_np(x)
    if isinstance(x, (list, tuple)):
        return type(x)(to_np(v) for v in x)
    return x


def to_torch(x):
    """numpy array/scalar -> torch tensor (dtype preserved — float32 inputs stay
    float32 through numpy ops); tuples mapped; plain ints/floats pass through."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x))
    if isinstance(x, np.generic):
        return torch.tensor(x.item())
    if isinstance(x, tuple):
        return tuple(to_torch(v) for v in x)
    return x


def bridge(fn):
    """Wrap a numpy function so a torch test can call it: torch args in -> numpy,
    numpy result -> torch out."""
    def wrapped(*args, **kwargs):
        na = [to_np(a) for a in args]
        nk = {k: to_np(v) for k, v in kwargs.items()}
        out = fn(*na, **nk)
        return to_torch(out)
    return wrapped


def _compiled_test(pid, name):
    return os.path.join(_COMPILED, f"test_p{pid}_{name}.pyc")


def run_numpy(pid, name, numpy_source):
    """Exec the numpy source, bridge every top-level function it defines, expose
    them as the problem's module, and run the compiled torch test.
    Returns (passed: bool, error: str|None)."""
    if torch is None:
        return False, "torch unavailable (needed to drive the existing test)"
    pyc = _compiled_test(pid, name)
    if not os.path.exists(pyc):
        return False, f"no compiled test for p{pid}_{name}"
    # Build the user's numpy namespace.
    user_ns = {"__name__": f"np_p{pid}_{name}"}
    try:
        exec(compile(numpy_source, f"p{pid}_{name}.py", "exec"), user_ns)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    # Expose bridged callables as the module the test imports (`from pNNx_name import *`).
    mod = types.ModuleType(f"p{pid}_{name}")
    for k, v in user_ns.items():
        if callable(v) and not k.startswith("_"):
            setattr(mod, k, bridge(v))
        elif not k.startswith("__"):
            setattr(mod, k, v)
    if _PROBLEMS not in sys.path:
        sys.path.insert(0, _PROBLEMS)
    saved = sys.modules.get(f"p{pid}_{name}")
    sys.modules[f"p{pid}_{name}"] = mod
    try:
        loader = SourcelessFileLoader(f"test_p{pid}_{name}", pyc)
        spec = importlib.util.spec_from_loader(f"test_p{pid}_{name}", loader)
        tmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tmod)
        getattr(tmod, f"test_p{pid}_{name}")()
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if saved is not None:
            sys.modules[f"p{pid}_{name}"] = saved
        else:
            sys.modules.pop(f"p{pid}_{name}", None)
