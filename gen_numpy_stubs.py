"""Generate NumPy variant stubs in problems_numpy/ for every numpy-supported
problem (NUMPY_SUPPORTED), using the signature from its numpy reference.

  python gen_numpy_stubs.py        # idempotent; only writes pristine/new stubs
"""
import ast
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "problems_numpy")
sys.path.insert(0, HERE)
from lib.solutions_numpy import NUMPY_PARENTS, NUMPY_SUPPORTED  # noqa: E402

_RE = re.compile(r"^p(\d+[a-z]?)_(.+)\.py$")

STUB = '''"""
Problem {pid} (NumPy): {name}

Solve in NumPy. Graded against the same hidden test as the torch variant via the
torch<->numpy bridge.  Run:  uv run python check.py {pid} --numpy
"""
import numpy as np  # noqa: F401


def {sig}:
    raise NotImplementedError


if __name__ == "__main__":
    import os as _os, re as _re, sys as _sys
    _p = _os.path.abspath(__file__)
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_p)))
    from lib.np_bridge import run_numpy
    _m = _re.match(r"p(\\d+[a-z]?)_(.+)\\.py", _os.path.basename(_p))
    _ok, _err = run_numpy(_m.group(1), _m.group(2), open(_p).read())
    print("PASS" if _ok else f"FAIL: {{_err}}")
    raise SystemExit(0 if _ok else 1)
'''


def _is_pristine(path, name):
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = [n for n in node.body if not (isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant))]
            return (len(body) == 1 and isinstance(body[0], ast.Raise))
    return False


def _sig_of(src, name):
    """Signature 'name(args, ...)' from the numpy reference function def."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return f"{name}{ast.unparse(node.args).join(('(', ')'))}"
    return f"{name}(*args)"


def main():
    os.makedirs(OUT, exist_ok=True)
    # child id -> (name, parent)
    children = {}
    for f in sorted(glob.glob(os.path.join(HERE, "problems", "p*_*.py"))):
        m = _RE.match(os.path.basename(f))
        if m:
            pid, name = m.groups()
            children[pid] = (name, f"{int(re.match(r'[0-9]+', pid).group()):02d}")

    written = skipped = 0
    for pid in sorted(NUMPY_SUPPORTED):
        if pid not in children:
            continue
        name, parent = children[pid]
        src = NUMPY_PARENTS.get(parent)
        if not src:
            continue
        path = os.path.join(OUT, f"p{pid}_{name}.py")
        if os.path.exists(path) and not _is_pristine(path, name):
            skipped += 1  # learner has work here
            continue
        with open(path, "w") as f:
            f.write(STUB.format(pid=pid, name=name, sig=_sig_of(src, name)))
        written += 1
    print(f"numpy stubs: {written} written, {skipped} preserved (had work)")


if __name__ == "__main__":
    main()
