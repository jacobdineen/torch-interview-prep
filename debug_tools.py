"""Helpers powering check.py's --hint / --explain / --solution / --time modes."""
import ast
import glob
import json
import os
import re
import subprocess
import sys
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
HINT_STATE_FILE = os.path.join(HERE, ".hint_state.json")
PROGRESS_FILE = os.path.join(HERE, ".progress.json")
SOLUTION_UNLOCK_FILE = os.path.join(HERE, ".solution_unlock.json")
NOTES_FILE = os.path.join(HERE, ".notes.json")


# -------------------------- hint state --------------------------

def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    except Exception:
        pass


# -------------------------- notes --------------------------

def add_note(pid, text):
    """Append a free-text note for a problem; resurfaces under --explain."""
    notes = _load_json(NOTES_FILE)
    notes.setdefault(pid, []).append(text)
    _save_json(NOTES_FILE, notes)
    print(f"  Noted for {pid}: {text}")


def get_notes(pid):
    return _load_json(NOTES_FILE).get(pid, [])


# -------------------------- hints --------------------------

def show_hints(pid, reset=False):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        from hints import get_hints
    except ImportError:
        print("  (hints.py not found)")
        return
    hints = get_hints(pid)
    if not hints:
        print(f"  No hints available for {pid}.")
        return
    state = _load_json(HINT_STATE_FILE)
    if reset:
        state.pop(pid, None)
        _save_json(HINT_STATE_FILE, state)
        print(f"  Hint counter for {pid} reset.")
        return
    used = state.get(pid, 0)
    if used >= len(hints):
        print(f"  All {len(hints)} hint(s) revealed for {pid}.")
        print(f"  Re-run with --reset-hints to start over, or try --explain / --solution.")
        return
    hint = hints[used]
    print()
    print(f"  Hint {used + 1}/{len(hints)} for problem {pid}:")
    for line in textwrap.wrap(hint, width=78,
                               initial_indent="    ", subsequent_indent="    "):
        print(line)
    state[pid] = used + 1
    _save_json(HINT_STATE_FILE, state)
    if used + 1 < len(hints):
        print(f"\n  (run again with --hint for the next hint; {len(hints) - used - 1} left)")


# -------------------------- explain --------------------------

def show_explain(pid, problem_path):
    """Describe what the test for this problem checks, in plain English."""
    src = open(problem_path).read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        print(f"  Cannot parse {os.path.basename(problem_path)}: {e}")
        return
    doc = ast.get_docstring(tree) or ""
    # Symbols (functions / classes) defined at top level — these are what the test exercises.
    symbols = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = ", ".join(a.arg for a in node.args.args)
            symbols.append(("def", f"{node.name}({args})"))
        elif isinstance(node, ast.AsyncFunctionDef):
            args = ", ".join(a.arg for a in node.args.args)
            symbols.append(("async def", f"{node.name}({args})"))
        elif isinstance(node, ast.ClassDef):
            symbols.append(("class", node.name))

    # Step label is the part after the colon in line 2 of the docstring.
    step_label = ""
    if doc:
        first_line = doc.strip().split("\n", 1)[0]
        m = re.match(r"Problem\s+\S+:\s*(.+)", first_line)
        if m:
            step_label = m.group(1).strip()

    print()
    print(f"  Problem {pid}:")
    if step_label:
        print(f"    Verifies: {step_label}")
    if symbols:
        print(f"    You implement:")
        for kind, sig in symbols:
            print(f"      {kind} {sig}")
    # A worked example (input -> output), if one can be extracted without giving
    # away the solution.
    try:
        from examples import example_for
        slug = os.path.basename(problem_path)[len(f"p{pid}_"):-3]
        ex = example_for(pid, slug)
    except Exception:
        ex = None
    if ex:
        print(f"    Example:")
        for s in ex.get("setup", []):
            print(f"      Input:   {s}")
        print(f"      Call:    {ex['call']}")
        if ex.get("output"):
            print(f"      Output:  {ex['output']}")
        elif ex.get("matches"):
            print(f"      Output:  should match {ex['matches']}")
    if doc:
        rest = doc.strip().split("\n", 2)
        if len(rest) > 2:
            extra = rest[2].strip()
            # The stub template leaves a bookkeeping line like
            #   (Split from parent problem 04: Problem 04: Broadcasting and Arithmetic)
            # Reformat it into a clean "Part of:" line rather than printing the
            # internal artifact as teaching prose.
            split_m = re.search(
                r"Split from parent problem\s+\d+:\s*(?:Problem\s+\d+:\s*)?(.+?)\)?\s*$",
                extra)
            if split_m:
                print(f"    Part of: {split_m.group(1).strip()}")
            elif extra:
                print(f"    Parent problem context:")
                for line in textwrap.wrap(extra, width=78,
                                           initial_indent="      ",
                                           subsequent_indent="      "):
                    print(line)
    # Concept blurb if available.
    try:
        from concepts import get_concept
        c = get_concept(pid)
        if c:
            print(f"    Concept:")
            for line in textwrap.wrap(c, width=78,
                                       initial_indent="      ",
                                       subsequent_indent="      "):
                print(line)
    except ImportError:
        pass

    # Any notes the user jotted with --note.
    notes = get_notes(pid)
    if notes:
        print(f"    Your notes:")
        for n in notes:
            for line in textwrap.wrap(n, width=78,
                                       initial_indent="      - ",
                                       subsequent_indent="        "):
                print(line)


# -------------------------- solution gating --------------------------

def _can_show_solution(pid):
    """Return (ok, reason). Solution shows iff:
       * user has ever_passed this problem, OR
       * user has explicitly unlocked it via --i-give-up."""
    progress = _load_json(PROGRESS_FILE)
    if progress.get(pid, {}).get("ever_passed"):
        return True, "you've already passed this problem"
    unlocks = _load_json(SOLUTION_UNLOCK_FILE)
    if unlocks.get(pid):
        return True, "you unlocked this with --i-give-up"
    return False, ""


def _record_unlock(pid):
    unlocks = _load_json(SOLUTION_UNLOCK_FILE)
    unlocks[pid] = True
    _save_json(SOLUTION_UNLOCK_FILE, unlocks)


def show_solution(pid, problem_path, i_give_up=False):
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        from solutions import get_solution
    except ImportError:
        print("  (solutions.py not found)")
        return

    ok, reason = _can_show_solution(pid)
    if not ok and not i_give_up:
        print()
        print(f"  Solution for {pid} is locked.")
        print(f"  Earn the unlock by either:")
        print(f"    1. Passing the problem at least once, then re-run --solution.")
        print(f"    2. Re-run with --solution --i-give-up to unlock it manually.")
        return
    if i_give_up and not ok:
        _record_unlock(pid)
        reason = "manually unlocked via --i-give-up"

    sol = get_solution(pid)
    if not sol:
        print()
        print(f"  No reference solution stored for problem {pid}.")
        print(f"  (Reference solutions live in solutions.py — add one if you'd like.)")
        return

    print()
    print(f"  Reference solution for {pid} ({reason}):")
    print()
    for line in sol.rstrip().split("\n"):
        print(f"    {line}")


# -------------------------- timing / benchmark --------------------------

def show_time(pid, problem_path):
    """Run user's impl and reference repeatedly; report median ms + speed ratio."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    try:
        from solutions import get_solution, get_benchmark
    except ImportError:
        print("  (solutions.py not found)")
        return

    sol = get_solution(pid)
    bench = get_benchmark(pid)
    if not sol or not bench:
        print()
        print(f"  No benchmark configured for problem {pid}.")
        print(f"  (Add a `BENCHMARKS[\"{pid}\"]` entry in solutions.py — see existing ones.)")
        return

    # Run the user's implementation N times.
    print()
    print(f"  Benchmarking problem {pid} ...")

    try:
        ns_user = {"__file__": problem_path}
        with open(problem_path) as f:
            exec(compile(f.read(), problem_path, "exec"), ns_user)
    except Exception as e:
        print(f"  Could not load user impl: {type(e).__name__}: {e}")
        return

    try:
        ns_ref = {}
        exec(compile(sol, "<reference>", "exec"), ns_ref)
    except Exception as e:
        print(f"  Could not load reference impl: {type(e).__name__}: {e}")
        return

    fn_name, setup_fn = bench["fn"], bench["setup"]
    try:
        user_fn = ns_user[fn_name]
        ref_fn = ns_ref[fn_name]
    except KeyError as kerr:
        print(f"  Benchmark looks for symbol {kerr} — missing from one of the impls.")
        return

    n_runs = bench.get("n_runs", 25)

    def _time(fn):
        times = []
        for _ in range(n_runs):
            args = setup_fn()
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        times.sort()
        return times[len(times) // 2]  # median

    user_t = _time(user_fn)
    ref_t = _time(ref_fn)
    ratio = user_t / ref_t if ref_t > 0 else float("inf")

    print(f"    User impl:       {user_t * 1e3:8.3f} ms (median of {n_runs})")
    print(f"    Reference impl:  {ref_t * 1e3:8.3f} ms (median of {n_runs})")
    if ratio < 1.1:
        verdict = "matches reference (within 10%)"
    elif ratio < 2.0:
        verdict = "a bit slower than reference (1-2x)"
    elif ratio < 10.0:
        verdict = f"~{ratio:.1f}x slower than reference"
    else:
        verdict = f"{ratio:.1f}x slower — likely an algorithmic issue"
    print(f"    Speed ratio:     {ratio:.2f}x  ({verdict})")
