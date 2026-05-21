"""Reset a problem (or all of them) back to its pristine stub.

  python reset.py 04a            # reset just one task (p04a_*)
  python reset.py 04             # reset every task in parent 04 (p04a, p04b, ...)
  python reset.py 04a --yes      # skip the confirmation prompt
  python reset.py 04 11 23a      # mix-and-match
  python reset.py --all          # reset everything (requires --yes too)
  python reset.py --all --yes
  python reset.py --status       # show which problems differ from their stub

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

_ID_RE = re.compile(r"^p(\d+[a-z]?)_")


def _check_stubs_exist():
    if not os.path.isdir(STUBS):
        print(f"ERROR: snapshot directory not found at {STUBS}")
        sys.exit(2)


def _id_from_path(path):
    m = _ID_RE.match(os.path.basename(path))
    return m.group(1) if m else None


def _stub_for(problem_path):
    return os.path.join(STUBS, os.path.basename(problem_path))


def _resolve(arg):
    """Resolve a CLI arg like '04' or '04a' to a list of matching problem paths."""
    m = re.match(r"^(\d+)([a-z]?)$", arg)
    if not m:
        return []
    num_int, letter = m.groups()
    num = f"{int(num_int):02d}"
    if letter:
        return sorted(glob.glob(os.path.join(PROBLEMS, f"p{num}{letter}_*.py")))
    return sorted(glob.glob(os.path.join(PROBLEMS, f"p{num}[a-z]_*.py")))


def _is_super_init(stmt):
    """True if stmt is exactly `super().__init__(...)`."""
    import ast
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "__init__"
        and isinstance(stmt.value.func.value, ast.Call)
        and isinstance(stmt.value.func.value.func, ast.Name)
        and stmt.value.func.value.func.id == "super"
    )


def _body_is_stub(body):
    """A function body is 'stub' if, after stripping docstring + Pass +
    `super().__init__()` calls, the remaining body is exactly
    `raise NotImplementedError`."""
    import ast
    filtered = []
    for i, stmt in enumerate(body):
        # Leading docstring
        if (i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            continue
        if isinstance(stmt, ast.Pass):
            continue
        if _is_super_init(stmt):
            continue
        filtered.append(stmt)
    if len(filtered) != 1 or not isinstance(filtered[0], ast.Raise):
        return False
    exc = filtered[0].exc
    if isinstance(exc, ast.Name):
        return exc.id == "NotImplementedError"
    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
        return exc.func.id == "NotImplementedError"
    return False


def _is_pristine(src):
    """A problem file is pristine if every top-level function and every class
    method has a stub body."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _body_is_stub(node.body):
                return False
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not _body_is_stub(m.body):
                        return False
    return True


def _diff_status():
    """List problems where the user has written something other than a pure stub."""
    edited = []
    for problem_path in sorted(glob.glob(os.path.join(PROBLEMS, "p*_*.py"))):
        with open(problem_path) as f:
            src = f.read()
        if not _is_pristine(src):
            edited.append(os.path.basename(problem_path))
    print()
    if edited:
        print(f"  {len(edited)} problem(s) edited:")
        for c in edited:
            print(f"    {c}")
    else:
        print("  No problems have been edited — everything is at its stub.")


def _confirm(msg):
    print(f"{msg} [y/N]: ", end="", flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer in ("y", "yes")


def _clear_progress_for(ids):
    if not os.path.exists(PROGRESS_FILE):
        return
    try:
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    except Exception:
        return
    for n in ids:
        progress.pop(n, None)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2, sort_keys=True)


def _reset(paths, force):
    pairs = []
    for p in paths:
        s = _stub_for(p)
        if not os.path.exists(s):
            print(f"  no snapshot for {os.path.basename(p)} — skipping")
            continue
        pairs.append((p, s))
    if not pairs:
        print("  nothing to do.")
        return
    print(f"  Will reset {len(pairs)} problem(s):")
    for p, _ in pairs:
        print(f"    {os.path.basename(p)}")
    if not force and not _confirm("  Proceed?"):
        print("  Aborted.")
        return
    for p, s in pairs:
        shutil.copy(s, p)
    _clear_progress_for([_id_from_path(p) for p, _ in pairs if _id_from_path(p)])
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
        all_paths = sorted(glob.glob(os.path.join(PROBLEMS, "p*_*.py")))
        _reset(all_paths, force=True)
        return
    paths = []
    for a in args:
        ps = _resolve(a)
        if not ps:
            print(f"  no problem found for {a!r}")
            continue
        paths.extend(ps)
    if not paths:
        print(__doc__)
        return
    _reset(paths, force=force)


if __name__ == "__main__":
    main()
