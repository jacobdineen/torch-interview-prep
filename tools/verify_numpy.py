"""Verify every numpy-supported problem: its numpy reference passes the existing
torch test through the bridge. This is the gate that the numpy variants actually
work (and the torch test is framework-neutral enough to grade them).

  python verify_numpy.py        # -> "numpy: N pass, M fail"; exit 1 on any fail
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from lib.np_bridge import run_numpy  # noqa: E402
from lib.solutions_numpy import NUMPY_PARENTS, NUMPY_SUPPORTED  # noqa: E402

_RE = re.compile(r"^p(\d+[a-z]?)_(.+)\.py$")


def main():
    # Map child id -> (name, parent) from the problems dir.
    children = {}
    for f in sorted(glob.glob(os.path.join(HERE, "problems", "p*_*.py"))):
        m = _RE.match(os.path.basename(f))
        if m:
            pid, name = m.groups()
            children[pid] = (name, f"{int(re.match(r'[0-9]+', pid).group()):02d}")

    npass = nfail = 0
    failures = []
    for pid in sorted(NUMPY_SUPPORTED):
        if pid not in children:
            failures.append(f"{pid}: claimed numpy-supported but no such problem")
            nfail += 1
            continue
        name, parent = children[pid]
        src = NUMPY_PARENTS.get(parent)
        if not src:
            failures.append(f"{pid} {name}: no numpy reference for parent {parent}")
            nfail += 1
            continue
        ok, err = run_numpy(pid, name, src)
        if ok:
            npass += 1
        else:
            nfail += 1
            failures.append(f"{pid} {name}: {err}")

    print(f"numpy: {npass} pass, {nfail} fail  ({len(NUMPY_SUPPORTED)} problems claim numpy support)")
    for fl in failures[:30]:
        print("  FAIL", fl)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
