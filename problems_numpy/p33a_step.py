"""
Problem 33a (NumPy): step

Solve in NumPy. Graded against the same hidden test as the torch variant via the
torch<->numpy bridge.  Run:  uv run python check.py 33a --numpy
"""
import numpy as np  # noqa: F401


def step(*args):
    raise NotImplementedError


if __name__ == "__main__":
    import os as _os, re as _re, sys as _sys
    _p = _os.path.abspath(__file__)
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_p)))
    from lib.np_bridge import run_numpy
    _m = _re.match(r"p(\d+[a-z]?)_(.+)\.py", _os.path.basename(_p))
    _ok, _err = run_numpy(_m.group(1), _m.group(2), open(_p).read())
    print("PASS" if _ok else f"FAIL: {_err}")
    raise SystemExit(0 if _ok else 1)
