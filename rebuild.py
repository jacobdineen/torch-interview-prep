"""Recompile every test source in tests/_src/ to .pyc in tests/_compiled/.

Each .py source is passed through `assert_rewriter.rewrite_assertions` so that
failed assertions surface the source line and the actual operand values in
their error message.
"""
import glob
import importlib.util
import marshal
import os
import sys

PREP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PREP, "tests", "_src")
OUT = os.path.join(PREP, "tests", "_compiled")
os.makedirs(OUT, exist_ok=True)

# Make assert_rewriter importable when this script is run directly.
if PREP not in sys.path:
    sys.path.insert(0, PREP)
from assert_rewriter import rewrite_assertions  # noqa: E402

count = 0
errors = []
for path in sorted(glob.glob(os.path.join(SRC, "test_p*.py"))):
    name = os.path.basename(path)[:-3]
    src = open(path).read()
    try:
        rewritten = rewrite_assertions(src)
        code = compile(rewritten, path, "exec")
    except Exception as e:
        errors.append((name, str(e)))
        continue
    pyc = os.path.join(OUT, name + ".pyc")
    with open(pyc, "wb") as f:
        f.write(importlib.util.MAGIC_NUMBER)
        f.write(b"\0" * 12)
        marshal.dump(code, f)
    count += 1

print(f"Recompiled {count} test files.")
if errors:
    print(f"\n{len(errors)} files failed:")
    for name, msg in errors:
        print(f"  {name}: {msg}")
