"""Convenience runner with debug helpers.

  python check.py 02a              # run the test for problem 02a
  python check.py 02               # run all of parent 02's children
  python check.py --next           # print the next unsolved problem (id + path)

Debug modes (don't run the test, show info instead):
  python check.py 02a --hint           # reveal the next graduated hint
  python check.py 02a --reset-hints    # reset hint counter for this problem
  python check.py 02a --explain        # describe what the test is checking
  python check.py 02a --solution       # show the reference implementation
                                       #   (gated; requires --i-give-up to first see)
  python check.py 02a --time           # benchmark your impl vs the reference
  python check.py 02a --note "TEXT"    # jot a note; shown later under --explain
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


def _next_unsolved(after=None):
    """Next problem not yet ever-passed, in curriculum order: (pid, path, wrapped).

    If ``after`` is given (a problem id like '07b'), return the first unsolved
    problem AFTER it. If nothing remains ahead, fall back to the earliest unsolved
    (an earlier gap) and set wrapped=True so the caller can say so.
    """
    import json
    from curriculum import all_problem_ids
    progress = {}
    pf = os.path.join(HERE, ".progress.json")
    if os.path.exists(pf):
        try:
            with open(pf) as f:
                progress = json.load(f)
        except Exception:
            progress = {}

    ids = all_problem_ids()

    def has_test(pid):
        # A problem with no compiled/source test can never be graded as passing,
        # so it must not be proposed as "next" (else --next dead-ends on it).
        return bool(glob.glob(os.path.join(HERE, "tests", "_compiled", f"test_p{pid}_*.pyc"))
                    or glob.glob(os.path.join(HERE, "tests", "_src", f"test_p{pid}_*.py")))

    def unsolved(pid):
        return has_test(pid) and not progress.get(pid, {}).get("ever_passed")

    def path_for(pid):
        matches = sorted(glob.glob(os.path.join(HERE, "problems", f"p{pid}_*.py")))
        return matches[0] if matches else None

    if after:
        m = re.match(r"^(\d+)([a-z]?)$", after)
        norm = f"{int(m.group(1)):02d}{m.group(2)}" if m else after
        if norm in ids:
            i = ids.index(norm)
            fwd = next((p for p in ids[i + 1:] if unsolved(p)), None)
            if fwd:
                return fwd, path_for(fwd), False

    earliest = next((p for p in ids if unsolved(p)), None)
    if earliest is None:
        return None, None, False
    return earliest, path_for(earliest), bool(after)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("id", nargs="?", help="problem id, e.g. '02' (parent) or '02a' (single task)")
    parser.add_argument("--next", action="store_true",
                        help="print the next unsolved problem (id, then its path) and exit")
    parser.add_argument("--note", metavar="TEXT",
                        help="save a free-text note for this problem (shown later under --explain)")
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

    if args.next:
        pid, path, wrapped = _next_unsolved(after=args.id)
        if not pid:
            print("All problems solved — nice work.")
            return
        if wrapped:
            print(f"Nothing left ahead — earlier gap: {pid}")
        else:
            print(f"Next unsolved: {pid}")
        if path:
            print(path)  # last line = path, for editors to open
        return

    if args.id is None:
        parser.error("a problem id is required (e.g. '02a'), or use --next")

    pid, matches = _resolve(args.id)
    if pid is None:
        print(f"unrecognized id {args.id!r}; expected e.g. 02 or 02a")
        sys.exit(2)
    if not matches:
        print(f"no problem found for {args.id!r}")
        sys.exit(2)

    if args.note is not None:
        from debug_tools import add_note
        for m in matches:
            add_note(_id_from_path(m), args.note)
        return

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
