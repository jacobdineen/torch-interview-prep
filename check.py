"""Convenience runner.

  python check.py 02a       # run problem 02a only
  python check.py 02        # run all of parent 02's children (02a, 02b, ...) in order

Same effect as `python problems/p02a_<topic>.py` but you don't have to type the topic.
"""
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) != 2:
        print("usage: python check.py <id>     e.g. 02 or 02a")
        sys.exit(2)
    arg = sys.argv[1]
    m = re.match(r"^(\d+)([a-z]?)$", arg)
    if not m:
        print(f"unrecognized {arg!r}; expected e.g. 02 or 02a")
        sys.exit(2)
    num_int, letter = m.groups()
    num = f"{int(num_int):02d}"

    if letter:
        matches = sorted(glob.glob(os.path.join(HERE, "problems", f"p{num}{letter}_*.py")))
    else:
        matches = sorted(glob.glob(os.path.join(HERE, "problems", f"p{num}[a-z]_*.py")))

    if not matches:
        print(f"no problem found for {arg!r}")
        sys.exit(2)

    if len(matches) == 1:
        os.execv(sys.executable, [sys.executable, matches[0]])

    rc = 0
    for f in matches:
        result = subprocess.run([sys.executable, f])
        if result.returncode != 0:
            rc = result.returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
