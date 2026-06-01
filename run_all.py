"""Batch runner and progress dashboard.

  python run_all.py              # run every problem (all 313)
  python run_all.py 4            # only parent 4's children (04a, 04b, 04c, 04d)
  python run_all.py 30-50        # parents 30 through 50 (inclusive)
  python run_all.py 4 7 12       # specific parents
  python run_all.py --status     # progress dashboard from .progress.json (no re-runs)
  python run_all.py --help / -h  # this summary
"""
import glob
import json
import os
import re
import subprocess
import sys

PREP = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = sorted(glob.glob(os.path.join(PREP, "problems", "p*_*.py")))
PROGRESS_FILE = os.path.join(PREP, ".progress.json")

# Shared tier definitions live in curriculum.py.
if PREP not in sys.path:
    sys.path.insert(0, PREP)
from curriculum import TIERS, tier_members, has_test  # noqa: E402


def _parse_filter(args):
    """Args are parent integers (e.g. '04') or ranges ('30-50').
    Filter selects any problem ID whose parent matches."""
    if not args:
        return lambda _id: True
    wanted_parents = set()
    for a in args:
        if "-" in a:
            lo, hi = a.split("-")
            for n in range(int(lo), int(hi) + 1):
                wanted_parents.add(n)
        else:
            wanted_parents.add(int(a))

    def keep(problem_id):
        m = re.match(r"^(\d+)", str(problem_id))
        return bool(m) and int(m.group(1)) in wanted_parents

    return keep


def _load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _bar(passed, total, width=18):
    filled = int(round(width * passed / total)) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def show_status():
    """Print a per-tier dashboard from cached .progress.json. No tests are run."""
    progress = _load_progress()
    nums_solved = {k for k, v in progress.items() if v.get("ever_passed")}
    total_passed = 0
    total = 0
    print()
    for name, lo, hi in TIERS:
        members = tier_members(lo, hi)
        solved = sum(1 for m in members if m in nums_solved)
        total_passed += solved
        total += len(members)
        bar = _bar(solved, len(members))
        unsolved = [m for m in members if m not in nums_solved]
        unsolved_str = f"  next: {unsolved[0]}" if unsolved else "  (complete)"
        print(f"  {bar} {solved:>3}/{len(members):<3} {name:<28}{unsolved_str}")
    print()
    print(f"  Overall: {total_passed}/{total} solved")


def main():
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print(__doc__)
        return
    if "--status" in args:
        show_status()
        return

    keep = _parse_filter(args)
    passed, failed, skipped = [], [], []
    for p in PROBLEMS:
        m = re.match(r"^p(\d+[a-z]?)_", os.path.basename(p))
        if not m: continue
        num = m.group(1)
        if not keep(num):
            continue
        if not has_test(num):
            # A test-less stub (e.g. a bare-parent left after a split) can't be
            # graded; skip it rather than counting a spurious fail.
            skipped.append(num)
            continue
        r = subprocess.run([sys.executable, p], capture_output=True, text=True)
        # The runner always prints PASS/FAIL as its FIRST line; everything after is
        # debug detail. Show only the headline here so the summary stays scannable.
        headline = (r.stdout or "").strip().split("\n")[0] if r.stdout else ""
        print(headline)
        (passed if r.returncode == 0 else failed).append(num)
    print()
    print(f"==== {len(passed)}/{len(passed) + len(failed)} passed ====")
    if failed:
        print("Failed:", " ".join(failed))
    if skipped:
        print("Skipped (no test):", " ".join(skipped))


if __name__ == "__main__":
    main()
