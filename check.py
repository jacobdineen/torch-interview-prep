"""Convenience runner with debug helpers.

  python check.py 02a              # run the test for problem 02a
  python check.py 02               # run all of parent 02's children

Debug modes (don't run the test, show info instead):
  python check.py 02a --hint           # reveal the next graduated hint
  python check.py 02a --reset-hints    # reset hint counter for this problem
  python check.py 02a --explain        # describe what the test is checking
  python check.py 02a --solution       # show the reference implementation
                                       #   (gated; requires --i-give-up to first see)
  python check.py 02a --time           # benchmark your impl vs the reference
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _resolve(arg):
    m = re.match(r"^(\d+)([a-z]?)$", arg)
    if not m:
        return None, None
    num_int, letter = m.groups()
    num = f"{int(num_int):02d}"
    if letter:
        matches = sorted(glob.glob(os.path.join(HERE, "problems", f"p{num}{letter}_*.py")))
    else:
        matches = sorted(glob.glob(os.path.join(HERE, "problems", f"p{num}[a-z]_*.py")))
    return f"{num}{letter}", matches


def _run(matches):
    if len(matches) == 1:
        os.execv(sys.executable, [sys.executable, matches[0]])
    rc = 0
    for f in matches:
        result = subprocess.run([sys.executable, f])
        if result.returncode != 0:
            rc = result.returncode
    sys.exit(rc)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("id", help="problem id, e.g. '02' (parent) or '02a' (single task)")
    parser.add_argument("--hint", action="store_true", help="reveal the next hint")
    parser.add_argument("--reset-hints", action="store_true",
                        help="reset hint counter so --hint starts from the beginning")
    parser.add_argument("--explain", action="store_true",
                        help="describe what this problem's test is checking")
    parser.add_argument("--solution", action="store_true",
                        help="show the reference implementation (requires --i-give-up first)")
    parser.add_argument("--i-give-up", action="store_true",
                        help="unlock --solution for this problem")
    parser.add_argument("--time", action="store_true",
                        help="time your implementation against the reference")
    args = parser.parse_args()

    pid, matches = _resolve(args.id)
    if pid is None:
        print(f"unrecognized id {args.id!r}; expected e.g. 02 or 02a")
        sys.exit(2)
    if not matches:
        print(f"no problem found for {args.id!r}")
        sys.exit(2)

    if args.hint or args.reset_hints:
        from debug_tools import show_hints
        for m in matches:
            show_hints(_id_from_path(m), reset=args.reset_hints)
        return
    if args.explain:
        from debug_tools import show_explain
        for m in matches:
            show_explain(_id_from_path(m), m)
        return
    if args.solution:
        from debug_tools import show_solution
        for m in matches:
            show_solution(_id_from_path(m), m, i_give_up=args.i_give_up)
        return
    if args.time:
        from debug_tools import show_time
        for m in matches:
            show_time(_id_from_path(m), m)
        return

    _run(matches)


def _id_from_path(path):
    m = re.match(r"^p(\d+[a-z]?)_", os.path.basename(path))
    return m.group(1) if m else None


if __name__ == "__main__":
    main()
