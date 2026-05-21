"""Run every problem stub and summarize how many pass.

  python run_all.py            # run all 77
  python run_all.py 30-50      # run problems 30 through 50 (inclusive)
  python run_all.py 65 67      # run specific problems
  python run_all.py --status   # show progress dashboard from .progress.json
                                 (no tests run; reads cached results)
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
from curriculum import TIERS, tier_members  # noqa: E402


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
    if "--status" in sys.argv[1:]:
        show_status()
        return

    keep = _parse_filter(sys.argv[1:])
    passed, failed = [], []
    for p in PROBLEMS:
        m = re.match(r"^p(\d+[a-z]?)_", os.path.basename(p))
        if not m: continue
        num = m.group(1)
        if not keep(num):
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


if __name__ == "__main__":
    main()
