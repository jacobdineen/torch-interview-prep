"""Run one step of a project (e.g. tiny-gpt-from-scratch).

Each step file defines one (or a few) function(s). We load the hidden reference
namespace, exec the user's step file with the reference functions injected into
its globals (so a step that calls earlier functions still works), then grade the
user's function with the step's hidden test. The user's own function shadows the
reference, so every step is tested in isolation.

Mirrors runner.py's contract: prints PASS/FAIL + a likely-cause line, supports
PREP_JSON=1 for the editor, records progress, and re-assembles solution.py on
PASS so the scaffold stays live.
"""
import importlib.util
import json
import os
import re
import sys
import textwrap
import time
import traceback
import types
from importlib.machinery import SourcelessFileLoader

_STEP_RE = re.compile(r"^(\d{4})_(.+)\.py$")


# ---------- locating the project ----------

def _find_project_root(step_path):
    d = os.path.dirname(os.path.abspath(step_path))
    while d and d != os.path.dirname(d):
        if os.path.exists(os.path.join(d, "project.json")):
            return d
        d = os.path.dirname(d)
    return None


def _stale_magic(pyc_path):
    """True if the .pyc was built by a different Python (its 4-byte magic header
    differs from this interpreter's). A committed .pyc after a Python upgrade
    otherwise loads as 'bad magic number', which reads like a wrong solution."""
    try:
        with open(pyc_path, "rb") as f:
            return f.read(4) != importlib.util.MAGIC_NUMBER
    except OSError:
        return True


def _ensure_compiled(project_root, force=False):
    """Compile the hidden _ref sources to _compiled/*.pyc if missing, stale, or
    built by a different Python, so the project works on a fresh checkout (and
    after a Python upgrade) without running the build script."""
    import py_compile
    ref_dir = os.path.join(project_root, "tests", "_ref")
    out_dir = os.path.join(project_root, "tests", "_compiled")
    os.makedirs(out_dir, exist_ok=True)
    for name in ("reference", "tests"):
        src = os.path.join(ref_dir, f"{name}.py")
        dst = os.path.join(out_dir, f"{name}.pyc")
        needs = (force or not os.path.exists(dst)
                 or os.path.getmtime(dst) < os.path.getmtime(src)
                 or _stale_magic(dst))
        if os.path.exists(src) and needs:
            try:
                py_compile.compile(src, cfile=dst, doraise=True)
            except Exception:
                pass


def _load_compiled(path, mod_name):
    loader = SourcelessFileLoader(mod_name, path)
    spec = importlib.util.spec_from_loader(mod_name, loader)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reference_funcs(project_root):
    pyc = os.path.join(project_root, "tests", "_compiled", "reference.pyc")
    mod = _load_compiled(pyc, "_proj_reference")
    return {k: v for k, v in vars(mod).items()
            if not k.startswith("_") and (
                isinstance(v, types.FunctionType)
                or (isinstance(v, type) and getattr(v, "__module__", "") == mod.__name__))}


# ---------- failure context (numpy-aware) ----------

def _user_fail_site(exc):
    """Deepest frame inside the project's steps/ dir, via the __cause__ chain."""
    best = (None, None, None, "")
    seen = set()
    e = exc
    needle = os.path.join("steps", "")
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        for f in traceback.extract_tb(e.__traceback__):
            if needle in f.filename or "/steps/" in f.filename:
                best = (f.filename, f.lineno, f.name, (f.line or ""))
        e = e.__cause__ or e.__context__
    return best


def _likely_cause(error_type, message):
    msg = (message or "").lower()
    if error_type == "NotImplementedError" or "notimplementederror" in msg:
        return "The function still raises NotImplementedError — fill in the body."
    if "gradient check failed" in msg:
        return ("Backward is wrong — the analytic gradient disagrees with the numeric "
                "one. Re-derive this op's local gradient and check the chain-rule factors.")
    if "shape mismatch" in msg:
        return ("Output SHAPE is wrong — check the axis you reduce/transpose over, "
                "keepdims, or a reshape.")
    if "crashed with" in msg:
        if any(w in msg for w in ("shape", "size", "dimension", "broadcast", "operands could not")):
            return "Your code raised a shape/broadcast error — check dims and reductions."
        if "key" in msg:
            return "A KeyError — a lookup id/char isn't in the dict you indexed."
        return "Your code raised an exception before returning — read the message above."
    if "values differ" in msg or "max abs diff" in msg:
        return ("Output VALUES are wrong — check the formula, the axis, an off-by-one, "
                "or a missing normalization term.")
    return None


def _print_failure_context(exc):
    ff, fl, fn_, line = _user_fail_site(exc)
    if ff:
        print("  In your code:")
        print(f"    {os.path.basename(ff)}:{fl} in {fn_}")
        if line:
            print(f"      {line}")
    cause = _likely_cause(type(exc).__name__, str(exc))
    if cause:
        print()
        print("  Likely cause:")
        for ln in textwrap.wrap(cause, width=78, initial_indent="    ", subsequent_indent="    "):
            print(ln)


# ---------- progress ----------

def _progress_file(project_root):
    return os.path.join(project_root, ".project_progress.json")


def _load_progress(project_root):
    pf = _progress_file(project_root)
    if not os.path.exists(pf):
        return {}
    try:
        with open(pf) as f:
            return json.load(f)
    except Exception:
        return {}


def _record(project_root, step_id, passed):
    from lib import store
    store.record_project(project_root, step_id, passed)


# ---------- progress display + manifest ----------

def _load_manifest(project_root):
    with open(os.path.join(project_root, "project.json")) as f:
        return json.load(f)


def _print_progress_after_pass(project_root, step_id):
    try:
        man = _load_manifest(project_root)
        steps = man["steps"]
        ever = {k for k, v in _load_progress(project_root).items() if v.get("ever_passed")}
        ids = [s["id"] for s in steps]
        by_id = {s["id"]: s for s in steps}
        part_idx = by_id[step_id]["part"]
        part = man["parts"][part_idx]
        members = [s["id"] for s in steps if s["part"] == part_idx]
        solved = sum(1 for m in members if m in ever)
        width = 18
        filled = int(round(width * solved / len(members))) if members else 0
        bar = "[" + "#" * filled + "-" * (width - filled) + "]"
        print()
        print("  Progress:")
        print(f"    {bar} {solved}/{len(members)}  Part {part_idx + 1}: {part['title']}")
        cur = ids.index(step_id)
        forward = next((i for i in ids[cur + 1:] if i not in ever), None)
        if forward:
            print(f"    next: {forward}  {by_id[forward]['name']}")
        else:
            print("    no later step unsolved — fill remaining gaps or run the scaffold")
        total = sum(1 for s in steps if s["id"] in ever)
        print(f"    overall: {total}/{len(steps)} steps")
    except Exception:
        pass


# ---------- JSON (editor) ----------

def _result_payload(step_id, name, step_path, status, error_type, message, exc):
    """Structured run-result dict (shared by the JSON emitter and the .last_run.json
    signal that lets the web UI react to runs started from nvim)."""
    out = {
        "status": status, "problem": step_id, "name": name,
        "file": os.path.abspath(step_path),
        "error_type": error_type, "message": message,
        "fail_file": None, "fail_line": None, "fail_func": None,
        "diff": None, "detail": None, "hint": _likely_cause(error_type, message),
    }
    if exc is not None:
        ff, fl, fn_, _ = _user_fail_site(exc)
        out["fail_file"], out["fail_line"], out["fail_func"] = ff, fl, fn_
    return out


def _emit_json(step_id, name, step_path, status, error_type, message, exc):
    print(json.dumps(_result_payload(step_id, name, step_path,
                                     status, error_type, message, exc)))


def _write_last_run(project_root, step_id, name, step_path, status, error_type, message, exc):
    """Persist the latest run result to the repo-root .last_run.json so the web UI
    can react to project-step runs started from nvim. Best-effort + atomic."""
    out = _result_payload(step_id, name, step_path,
                          status, error_type, message, exc)
    try:
        _name = _load_manifest(project_root).get("name") or os.path.basename(project_root)
    except Exception:
        _name = os.path.basename(project_root)
    out["key"] = f"proj:{_name}:{step_id}"
    out["ts"] = time.time()
    repo_root = os.path.dirname(os.path.dirname(project_root))
    path = os.path.join(repo_root, ".last_run.json")
    try:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, path)
    except OSError:
        pass


# ---------- main entry ----------

def run_step(step_path):
    base = os.path.basename(step_path)
    m = _STEP_RE.match(base)
    if not m:
        print(f"[project_runner] Not a step file: {base}")
        return 2
    step_id, name = m.groups()
    root = _find_project_root(step_path)
    if root is None:
        print(f"[project_runner] No project.json above {step_path}")
        return 2

    json_mode = os.environ.get("PREP_JSON") == "1"
    label = f"Step {step_id} ({name})"
    _ensure_compiled(root)

    def _load_refs_and_tests():
        funcs = _reference_funcs(root)
        mod = _load_compiled(
            os.path.join(root, "tests", "_compiled", "tests.pyc"), "_tgp_tests")
        return funcs, mod

    try:
        try:
            ref_funcs, tests_mod = _load_refs_and_tests()
        except ImportError:
            # Most likely a committed .pyc built by a different Python ("bad magic
            # number"); recompile against this interpreter and retry once.
            if not json_mode:
                print("[project_runner] recompiling project tests for your Python...")
            _ensure_compiled(root, force=True)
            ref_funcs, tests_mod = _load_refs_and_tests()
        test_fn = getattr(tests_mod, f"test_{step_id}_{name}", None)
        if test_fn is None:
            print(f"[project_runner] No test for {step_id}_{name}")
            return 2

        # Exec the user's step with reference functions in scope, then shadow.
        user_globals = {"__name__": f"step_{step_id}", "__file__": os.path.abspath(step_path)}
        user_globals.update(ref_funcs)
        with open(step_path) as f:
            src = f.read()
        exec(compile(src, step_path, "exec"), user_globals)
        ns = user_globals

        test_fn(ns)
    except NotImplementedError:
        _write_last_run(root, step_id, name, step_path, "fail", "NotImplementedError",
                        "the function still raises NotImplementedError", None)
        if json_mode:
            _emit_json(step_id, name, step_path, "fail", "NotImplementedError",
                       "the function still raises NotImplementedError", None)
        else:
            print(f"FAIL {label}: NotImplementedError — fill in the function body.")
        _record(root, step_id, False)
        return 1
    except AssertionError as e:
        msg = str(e) or "(no message)"
        _write_last_run(root, step_id, name, step_path, "fail", "AssertionError", msg, e)
        if json_mode:
            _emit_json(step_id, name, step_path, "fail", "AssertionError", msg, e)
        else:
            print(f"FAIL {label}")
            print(f"  AssertionError: {msg}")
            _print_failure_context(e)
        _record(root, step_id, False)
        return 1
    except Exception as e:
        _write_last_run(root, step_id, name, step_path, "fail", type(e).__name__, str(e), e)
        if json_mode:
            _emit_json(step_id, name, step_path, "fail", type(e).__name__, str(e), e)
        else:
            print(f"FAIL {label}: {type(e).__name__}: {e}")
            _print_failure_context(e)
        _record(root, step_id, False)
        return 1

    _record(root, step_id, True)
    _write_last_run(root, step_id, name, step_path, "pass", None, None, None)
    try:
        assemble_solution(root)
    except Exception as e:
        # Best-effort: the PASS still stands, but warn (to stderr, so it doesn't
        # corrupt the single JSON line the editor reads in PREP_JSON mode) so a
        # broken assembly doesn't silently leave the scaffold's solution.py stale.
        print(f"[project_runner] warning: could not re-assemble solution.py: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
    if json_mode:
        _emit_json(step_id, name, step_path, "pass", None, None, None)
    else:
        print(f"PASS {label}")
        _print_progress_after_pass(root, step_id)
    return 0


# ---------- live solution.py assembly ----------

def assemble_solution(project_root):
    """Concatenate the user's solved step functions into solution.py (what the
    scaffold imports). Unsolved steps appear as a TODO marker."""
    man = _load_manifest(project_root)
    ever = {k for k, v in _load_progress(project_root).items() if v.get("ever_passed")}
    parts = man["parts"]
    out = [f'"""{man.get("title", man["name"])} — assembled scaffold.',
           "This file is generated from your solved steps; edit the steps, not this.",
           '"""', "", "import numpy as np", "import torch", "import torch.nn as nn",
           "import torch.nn.functional as F", ""]
    cur_part = None
    for s in man["steps"]:
        sid, name, part = s["id"], s["name"], s["part"]
        if part != cur_part:
            cur_part = part
            out.append(f"# ===== Part {part + 1}: {parts[part]['title']} =====")
            out.append("")
        step_file = os.path.join(project_root, "steps", f"{sid}_{name}.py")
        body = _extract_solved_def(step_file) if sid in ever else None
        if body:
            out.append(f"# -- Step {sid}  {name} --")
            out.extend(body)
            out.append("")
        else:
            out.append(f"# -- Step {sid}  {name}  (TODO: solve) --")
            out.append("")
    sol = os.path.join(project_root, "solution.py")
    with open(sol, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")
    return sol


def _extract_solved_def(step_file):
    """Return the source lines of every top-level def/assignment in the step file
    except the __main__ guard and imports — i.e. the user's solved code."""
    import ast
    if not os.path.exists(step_file):
        return None
    with open(step_file) as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    lines = src.splitlines()
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = lines[node.lineno - 1: node.end_lineno]
            out.extend(seg)
            out.append("")
    return out or None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python project_runner.py <path/to/steps/NNNN_name.py>")
        sys.exit(2)
    sys.exit(run_step(sys.argv[1]))
