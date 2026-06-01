"""Verify every problem's reference solution still passes its compiled test.

For each problem child, the reference implementation (from solutions.py) is written
into a TEMPORARY directory placed first on sys.path, so the compiled test imports
the reference instead of the stub. Your working files in problems/ are never
touched — safe to run locally as well as in CI.

  python verify_problems.py        # -> "N pass, M fail, K skip"; exit 1 on any fail
"""
import glob
import importlib.util
import os
import re
import shutil
import sys
import tempfile
from importlib.machinery import SourcelessFileLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(HERE, "problems")
COMPILED = os.path.join(HERE, "tests", "_compiled")
sys.path.insert(0, HERE)

from solutions import PARENT_SOLUTIONS  # noqa: E402

PRELUDE = ("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
           "import numpy as np\n\n\n")
_RE = re.compile(r"^p(\d+[a-z]?)_(.+)\.py$")


def main():
    tmp = tempfile.mkdtemp(prefix="ref_problems_")
    sys.path.insert(0, tmp)  # resolve `from pNN_name import *` to the reference copy
    npass = nfail = 0
    failures = []
    skips = []
    try:
        pending = []
        for f in sorted(glob.glob(os.path.join(PROBLEMS, "p*_*.py"))):
            m = _RE.match(os.path.basename(f))
            if not m:
                continue
            pid, name = m.groups()
            parent = f"{int(re.match(r'[0-9]+', pid).group()):02d}"
            ref = PARENT_SOLUTIONS.get(parent)  # full parent (includes shared helpers)
            if not ref:
                skips.append(f"{pid} {name} (no reference in solutions.py)")
                continue
            with open(os.path.join(tmp, os.path.basename(f)), "w") as out:
                out.write(PRELUDE + ref + "\n")
            pending.append((pid, name))

        for pid, name in pending:
            mod_name = f"test_p{pid}_{name}"
            pyc = os.path.join(COMPILED, f"{mod_name}.pyc")
            if not os.path.exists(pyc):
                skips.append(f"{pid} {name} (no compiled test)")
                continue
            try:
                loader = SourcelessFileLoader(mod_name, pyc)
                spec = importlib.util.spec_from_loader(mod_name, loader)
                tmod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tmod)  # imports the reference stub from tmp
                getattr(tmod, mod_name)()
                npass += 1
            except Exception as e:
                nfail += 1
                failures.append(f"{pid} {name}: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"problems: {npass} pass, {nfail} fail, {len(skips)} skip")
    for fl in failures[:25]:
        print("  FAIL", fl)
    for sk in skips:
        print("  SKIP", sk)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
