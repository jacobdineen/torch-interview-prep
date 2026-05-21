"""Reset a problem (or all of them) back to its pristine stub.

  python reset.py 04            # reset problem 04 (asks for confirmation)
  python reset.py 04 --yes      # skip the confirmation prompt
  python reset.py 04 11 23      # reset multiple problems
  python reset.py --all         # reset every problem (requires --yes too)
  python reset.py --all --yes
  python reset.py --status      # show which problems differ from their stub

Also clears `.progress.json` entries for any reset problem so the dashboard
reflects that you're starting over on those.
"""
import glob
import json
import os
import re
import shutil
import sys

PREP = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(PREP, "problems")
STUBS = os.path.join(PREP, ".stubs")
PROGRESS_FILE = os.path.join(PREP, ".progress.json")


def _check_stubs_exist():
    if not os.path.isdir(STUBS):
        print(f"ERROR: snapshot directory not found at {STUBS}")
        print("(If you wiped it, ask Claude to re-run snapshot_stubs.py)")
        sys.exit(2)


def _match_problem(num_str):
    """Find the problem file matching a 2-digit problem number."""
    matches = glob.glob(os.path.join(PROBLEMS, f"p{num_str}_*.py"))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"ambiguous: multiple files match p{num_str}_*.py — {matches}")
        sys.exit(2)
    return matches[0]


def _stub_for(problem_path):
    """Find the matching stub in .stubs/ given a problem path."""
    return os.path.join(STUBS, os.path.basename(problem_path))


def _diff_status():
    """List problems whose current file differs from the snapshot."""
    changed, missing = [], []
    for problem_path in sorted(glob.glob(os.path.join(PROBLEMS, "p*_*.py"))):
        stub_path = _stub_for(problem_path)
        if not os.path.exists(stub_path):
            missing.append(os.path.basename(problem_path))
            continue
        with open(problem_path) as f: cur = f.read()
        with open(stub_path) as f: pristine = f.read()
        if cur != pristine:
            changed.append(os.path.basename(problem_path))
    print()
    if changed:
        print(f"  {len(changed)} problem(s) edited (differ from stub):")
        for c in changed:
            print(f"    {c}")
    else:
        print("  No problems have been edited — everything matches the stubs.")
    if missing:
        print()
        print(f"  WARNING: {len(missing)} problem(s) have no snapshot in .stubs/")


def _confirm(msg):
    """Prompt the user; return True if they confirm."""
    print(f"{msg} [y/N]: ", end="", flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer in ("y", "yes")


def _clear_progress_for(nums):
    if not os.path.exists(PROGRESS_FILE):
        return
    try:
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    except Exception:
        return
    for n in nums:
        progress.pop(n, None)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2, sort_keys=True)


def _reset(nums, force):
    """Reset the given list of 2-digit problem numbers."""
    pairs = []
    for n in nums:
        p = _match_problem(n)
        if p is None:
            print(f"  no problem found for {n!r}")
            continue
        s = _stub_for(p)
        if not os.path.exists(s):
            print(f"  no snapshot for {os.path.basename(p)} — skipping")
            continue
        pairs.append((n, p, s))
    if not pairs:
        print("  nothing to do.")
        return
    print(f"  Will reset {len(pairs)} problem(s):")
    for n, p, _ in pairs:
        print(f"    {os.path.basename(p)}")
    if not force and not _confirm("  Proceed?"):
        print("  Aborted.")
        return
    for n, p, s in pairs:
        shutil.copy(s, p)
    _clear_progress_for([n for n, _, _ in pairs])
    print(f"  Reset {len(pairs)} problem(s).")


def main():
    _check_stubs_exist()
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        return
    if "--status" in args:
        _diff_status()
        return
    force = "--yes" in args
    args = [a for a in args if a != "--yes"]
    if "--all" in args:
        if not force:
            print("--all requires --yes (this rewrites every problem file)")
            sys.exit(2)
        nums = sorted({
            re.match(r"p(\d+)_", os.path.basename(p)).group(1)
            for p in glob.glob(os.path.join(PROBLEMS, "p*_*.py"))
        })
        _reset(nums, force=True)
        return
    nums = []
    for a in args:
        if not a.isdigit():
            print(f"  ignoring {a!r}: expected a problem number")
            continue
        nums.append(f"{int(a):02d}")
    if not nums:
        print(__doc__)
        return
    _reset(nums, force=force)


if __name__ == "__main__":
    main()
