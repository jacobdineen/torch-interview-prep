"""Recompile every test source in tests/_src/ to .pyc in tests/_compiled/.

Run this if you ever modify a test source file (you shouldn't need to — peeking
at _src/ defeats the point — but it's here if you want to add your own tests).
"""
import os, glob, marshal, importlib.util

PREP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PREP, "tests", "_src")
OUT = os.path.join(PREP, "tests", "_compiled")
os.makedirs(OUT, exist_ok=True)

count = 0
for path in sorted(glob.glob(os.path.join(SRC, "test_p*.py"))):
    name = os.path.basename(path)[:-3]  # strip .py
    src = open(path).read()
    code = compile(src, path, "exec")
    pyc = os.path.join(OUT, name + ".pyc")
    with open(pyc, "wb") as f:
        f.write(importlib.util.MAGIC_NUMBER)
        f.write(b"\0" * 12)
        marshal.dump(code, f)
    count += 1
print(f"Recompiled {count} test files.")
