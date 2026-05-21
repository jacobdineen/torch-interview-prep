"""Convenience runner: `python check.py 02` -> runs problem 02.

Same effect as `python problems/p02_<topic>.py` but you don't have to type the topic.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) != 2:
        print("usage: python check.py <problem-number>   (e.g. python check.py 02)")
        sys.exit(2)
    arg = sys.argv[1]
    if not arg.isdigit():
        print(f"problem number must be an integer; got {arg!r}")
        sys.exit(2)
    num = f"{int(arg):02d}"
    matches = glob.glob(os.path.join(HERE, "problems", f"p{num}_*.py"))
    if not matches:
        print(f"no problem found matching p{num}_*.py in problems/")
        sys.exit(2)
    if len(matches) > 1:
        print(f"ambiguous: multiple files match p{num}_*.py — {matches}")
        sys.exit(2)
    os.execv(sys.executable, [sys.executable, matches[0]])


if __name__ == "__main__":
    main()
