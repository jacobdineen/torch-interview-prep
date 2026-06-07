"""Run a problem stub's compiled test and print PASS/FAIL with feedback.

Tutor-mode features:
  * Tensor-aware locals dump: shows shape, dtype, head/tail values.
  * Two-tensor diff: when the test bound an `actual`/`expected`-style pair, shows
    a focused shape + dtype + first-mismatch comparison.
  * Concept blurbs on PASS (from concepts.py).
  * Progress tracking: every PASS/FAIL is recorded in .progress.json so run_all.py
    can show a "X/77 solved" dashboard.
"""
import datetime
import importlib.util
import json
import os
import re
import sys
import textwrap
import time
import traceback
from importlib.machinery import SourcelessFileLoader

PREP = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_DIR = os.path.join(PREP, "problems")
COMPILED_DIR = os.path.join(PREP, "tests", "_compiled")
PROGRESS_FILE = os.path.join(PREP, ".progress.json")
LAST_RUN_FILE = os.path.join(PREP, ".last_run.json")

_TEST_FILE_RE = re.compile(r"test_p\d+_.+\.py$")
_TEST_FN_RE = re.compile(r"^test_p\d+_")


# ---------- value formatting ----------

def _fmt_tensor(t):
    """Compact, debug-friendly tensor representation."""
    import torch
    shape = tuple(t.shape)
    info = f"shape={shape}, dtype={t.dtype}"
    if t.numel() == 0:
        return f"Tensor({info}, empty)"
    if t.numel() <= 12:
        return f"Tensor({info}, data={t.tolist()})"
    # Larger tensor: show a small preview from the flat view.
    flat = t.detach().reshape(-1)
    head = flat[:4].tolist()
    tail = flat[-2:].tolist()
    return f"Tensor({info}, head={head}, tail={tail}, " \
           f"min={t.min().item():.4g}, max={t.max().item():.4g})"


def _short_repr(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return _fmt_tensor(v)
    except Exception:
        pass
    if isinstance(v, (int, float, bool, str, type(None))):
        return repr(v)
    if isinstance(v, (tuple, list)):
        if len(v) <= 8:
            return repr(v)
        return f"{type(v).__name__}(len={len(v)})"
    if isinstance(v, dict):
        if len(v) <= 6:
            return repr(v)
        return f"dict(len={len(v)}, keys={list(v.keys())[:5]}...)"
    return f"<{type(v).__name__}>"


# ---------- tensor diff ----------

def _diff_tensors(a, b, name_a="left", name_b="right"):
    """Return a multi-line string describing how two tensors differ. Empty string
    if they're equal (shouldn't happen at the failure site, but guards us)."""
    import torch
    out = []
    sa, sb = tuple(a.shape), tuple(b.shape)
    if sa != sb:
        out.append(f"      shape mismatch: {name_a}={sa}  vs  {name_b}={sb}")
        return "\n".join(out)
    if a.dtype != b.dtype:
        out.append(f"      dtype mismatch: {name_a}={a.dtype}  vs  {name_b}={b.dtype}")
    if not torch.is_floating_point(a):
        a_f = a.to(torch.float64)
        b_f = b.to(torch.float64)
    else:
        a_f, b_f = a, b
    diff = (a_f - b_f).abs()
    max_diff = diff.max().item() if diff.numel() else 0.0
    out.append(f"      max |{name_a} - {name_b}| = {max_diff:.6g}")
    # Find the first mismatching index (where diff exceeds a tiny tolerance).
    nz = (diff > max(1e-6, max_diff * 1e-6)).nonzero(as_tuple=False)
    if nz.numel() > 0:
        idx = tuple(nz[0].tolist())
        out.append(
            f"      first difference at index {idx}: "
            f"{name_a}={a_f[idx].item():.6g}  {name_b}={b_f[idx].item():.6g}"
        )
    return "\n".join(out)


def _looks_like_pair(name_a, name_b):
    """Heuristic: do these two local names look like an actual/expected pair?"""
    a, b = name_a.lower(), name_b.lower()
    pairs = [
        ("got", "expected"), ("actual", "expected"), ("expected", "got"),
        ("expected", "actual"), ("out", "expected"), ("expected", "out"),
        ("mine", "ref"), ("mine_out", "ref_out"), ("ref", "mine"),
    ]
    if (a, b) in pairs or (b, a) in pairs:
        return True
    # Also detect prefix conventions: "ref_*" vs "mine_*".
    if (a.startswith("ref") and b.startswith("mine")) or \
       (a.startswith("mine") and b.startswith("ref")):
        return True
    return False


def _print_tensor_pair_diffs(tensors):
    """If two locals are tensors and their names look like a pair, show the diff."""
    names = list(tensors.keys())
    shown = False
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _looks_like_pair(a, b):
                if not shown:
                    print("  Tensor comparison:")
                    shown = True
                print(f"    {a}  vs  {b}")
                print(_diff_tensors(tensors[a], tensors[b], a, b))


# ---------- traceback walking ----------

def _find_test_frame(tb):
    """Find the actual `test_p<NN>_<topic>` function frame (skip the helper
    `step()` frame which lives in the same file)."""
    while tb is not None:
        fn = tb.tb_frame.f_code.co_filename or ""
        cname = tb.tb_frame.f_code.co_name
        if _TEST_FILE_RE.search(os.path.basename(fn)) and _TEST_FN_RE.match(cname):
            return tb
        tb = tb.tb_next
    return None


def _user_fail_site(exc):
    """Deepest frame inside the user's problems/ file (the actual failing line),
    or (None, None, None, '') if the failure was raised inside test code. Walks the
    __cause__/__context__ chain so a RuntimeError that step() re-wraps as an
    AssertionError still points back at the user's line."""
    best = (None, None, None, "")
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        for f in traceback.extract_tb(e.__traceback__):
            if "problems" + os.sep in f.filename:
                best = (f.filename, f.lineno, f.name, (f.line or ""))
        e = e.__cause__ or e.__context__
    return best


def _const_scale(a, b):
    """If a ≈ k*b for a near-constant k (and k is not ~1), return k. Used to spot a
    missing/extra scaling term (e.g. /sqrt(d), mean-vs-sum). a, b are float tensors."""
    import torch
    mask = b.abs() > 1e-8
    if int(mask.sum().item()) < 4:
        return None
    r = a[mask] / b[mask]
    r = r[torch.isfinite(r)]
    if r.numel() < 4:
        return None
    mean = r.mean().item()
    std = r.std().item()
    if abs(mean) > 1e-6 and std / abs(mean) < 0.01 and abs(mean - 1.0) > 0.02:
        return mean
    return None


def _diff_struct(exc):
    """Structured description of the first actual/expected tensor pair at the test
    frame: {kind: shape|dtype|value, ...}. None if no such pair is bound."""
    tframe = _find_test_frame(exc.__traceback__)
    if tframe is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    locs = tframe.tb_frame.f_locals
    tensors = {k: v for k, v in locs.items()
               if not k.startswith("_") and isinstance(v, torch.Tensor)}
    names = list(tensors.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if not _looks_like_pair(a, b):
                continue
            ta, tb = tensors[a], tensors[b]
            if tuple(ta.shape) != tuple(tb.shape):
                return {"kind": "shape", "a": a, "b": b,
                        "shape_a": list(ta.shape), "shape_b": list(tb.shape)}
            if ta.dtype != tb.dtype:
                return {"kind": "dtype", "a": a, "b": b,
                        "dtype_a": str(ta.dtype), "dtype_b": str(tb.dtype)}
            af = ta.detach().to(torch.float64)
            bf = tb.detach().to(torch.float64)
            d = (af - bf).abs()
            max_diff = d.max().item() if d.numel() else 0.0
            first_index = None
            nz = (d > max(1e-6, max_diff * 1e-6)).nonzero(as_tuple=False)
            if nz.numel() > 0:
                first_index = tuple(nz[0].tolist())
            info = {"kind": "value", "a": a, "b": b,
                    "max_diff": max_diff, "first_index": first_index}
            scale = _const_scale(af, bf)
            if scale is not None:
                info["scale"] = scale
            return info
    return None


def _likely_cause(error_type, message, diff):
    """A single-sentence guess at what went wrong, for a learner. None if unsure."""
    msg = (message or "").lower()
    if error_type == "NotImplementedError":
        return "The function still raises NotImplementedError — fill in the body."
    if error_type and error_type != "AssertionError":
        # Runtime error raised inside the user's code.
        if any(w in msg for w in ("shape", "size", "dimension", "must match", "broadcast")):
            return ("Shape/dimension error at runtime — check broadcasting, a transpose, "
                    "or the reduction dim.")
        if "expected scalar type" in msg or "dtype" in msg or "same type" in msg:
            return "Dtype error — add a cast (.float(), .long(), .to(...))."
        if "device" in msg:
            return "Device mismatch — move tensors to the same device with .to(...)."
        return None
    if diff:
        if diff["kind"] == "shape":
            return ("Output SHAPE is wrong — likely reduced over the wrong dim, forgot "
                    "keepdim, or need a transpose/reshape.")
        if diff["kind"] == "dtype":
            return "Output DTYPE is wrong — add a cast (.float()/.long()/.to(...))."
        if diff["kind"] == "value":
            scale = diff.get("scale")
            if scale is not None:
                return (f"Values are off by a roughly constant factor (~{scale:.4g}x) — likely "
                        f"a missing/extra scaling term (e.g. /sqrt(d), /N, or mean vs sum).")
            return ("Output VALUES are wrong — check which axis you operate on, an off-by-one, "
                    "or a missing term in the formula.")
    # A runtime error raised inside the user's code, re-wrapped by step().
    if "crashed with" in msg:
        if any(w in msg for w in ("shape", "size", "dimension", "must match", "broadcast")):
            return ("Your code raised a shape/dimension error — check broadcasting, a "
                    "transpose, or the reduction dim.")
        if "expected scalar type" in msg or "dtype" in msg:
            return "Your code raised a dtype error — add a cast (.float()/.long()/.to(...))."
        if "device" in msg:
            return "Your code raised a device error — move tensors to the same device with .to(...)."
        return "Your code raised an exception before returning — read the error message above."
    # No named got/expected pair — fall back to the rewritten assertion message, which
    # encodes the failure kind (this is the common case for value/shape asserts).
    if "shape comparison" in msg or "shape mismatch" in msg:
        return ("Output SHAPE is wrong — likely reduced over the wrong dim, forgot "
                "keepdim, or need a transpose/reshape.")
    if "dtype mismatch" in msg:
        return "Output DTYPE is wrong — add a cast (.float()/.long()/.to(...))."
    if any(k in msg for k in ("max |left - right|", "first difference", "allclose", "isclose")):
        return ("Output VALUES are wrong — check which axis you operate on, an off-by-one, "
                "or a missing term in the formula.")
    return None


def _print_likely_cause(exc, diff=None):
    cause = _likely_cause(type(exc).__name__, str(exc),
                          diff if diff is not None else _diff_struct(exc))
    if cause:
        print()
        print("  Likely cause:")
        for line in textwrap.wrap(cause, width=78, initial_indent="    ", subsequent_indent="    "):
            print(line)


def _print_failure_context(exc):
    """Show user-code frames + tensor diff (if present) + relevant test locals."""
    # 1) User-code frames: point at the line in their .py file.
    user_tb = traceback.extract_tb(exc.__traceback__)
    user_frames = [f for f in user_tb if "problems" + os.sep in f.filename]
    if user_frames:
        print("  In your code:")
        for f in user_frames:
            print(f"    {os.path.basename(f.filename)}:{f.lineno} in {f.name}")
            if f.line:
                print(f"      {f.line}")

    # 2) Walk locals at the test frame.
    tframe = _find_test_frame(exc.__traceback__)
    if tframe is None:
        _print_likely_cause(exc)
        return
    locs = tframe.tb_frame.f_locals
    try:
        import torch
    except ImportError:
        torch = None

    keep = {}
    tensors = {}
    pair_names = set()
    for k, v in locs.items():
        if k.startswith("_"): continue
        if callable(v) and not hasattr(v, "shape"):
            continue
        if type(v).__name__ == "module": continue
        keep[k] = v
        if torch is not None and isinstance(v, torch.Tensor):
            tensors[k] = v

    # 3) Tensor pair diff — show FIRST, since this is usually the most actionable.
    if len(tensors) >= 2:
        names = list(tensors.keys())
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if _looks_like_pair(a, b):
                    pair_names.add(a); pair_names.add(b)
        if pair_names:
            print("  Tensor comparison:")
            shown_pairs = set()
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    if _looks_like_pair(a, b) and (a, b) not in shown_pairs:
                        shown_pairs.add((a, b))
                        print(f"    {a}  vs  {b}")
                        print(_diff_tensors(tensors[a], tensors[b], a, b))

    # 4) Locals — show pair tensors fully; for everything else, just shape+dtype.
    if keep:
        print("  Test locals when it failed:")
        # Pair tensors come first, with full preview.
        for k in pair_names:
            print(f"    {k} = {_short_repr(keep[k])}")
        # Other locals: compact summary line. Scalars / small things shown in full;
        # large tensors just show shape+dtype.
        for k, v in keep.items():
            if k in pair_names:
                continue
            if torch is not None and isinstance(v, torch.Tensor) and v.numel() > 12:
                # Compact: just shape + dtype, no values.
                print(f"    {k} = Tensor(shape={tuple(v.shape)}, dtype={v.dtype})")
            else:
                print(f"    {k} = {_short_repr(v)}")

    # 5) A one-line guess at the root cause — most useful for a learner.
    _print_likely_cause(exc)


# ---------- progress tracking ----------

def _load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _record(num, passed):
    # SQLite is the source of truth (atomic, race-free); it also re-exports
    # .progress.json so readers (web, check.py, nvim) keep working unchanged.
    from lib import store
    store.record_problem(num, passed)


# ---------- concept blurbs ----------

def _print_concept(num):
    try:
        if PREP not in sys.path:
            sys.path.insert(0, PREP)
        from lib.concepts import get_concept  # type: ignore
        text = get_concept(num)
        if not text:
            return
        print()
        print("  Concept:")
        for line in textwrap.wrap(text, width=78, initial_indent="    ", subsequent_indent="    "):
            print(line)
    except Exception:
        pass  # concept blurbs are best-effort


# ---------- per-PASS progress update ----------

def _print_progress_after_pass(num):
    """After a PASS, print this tier's bar + overall count for momentum."""
    try:
        if PREP not in sys.path:
            sys.path.insert(0, PREP)
        from lib.curriculum import find_tier, tier_members, TIERS, TOTAL_PROBLEMS  # type: ignore
        tier = find_tier(num)
        progress = _load_progress()
        ever = {k for k, v in progress.items() if v.get("ever_passed")}

        print()
        print("  Progress:")
        if tier is not None:
            tier_idx, tier_name, lo, hi = tier
            members = tier_members(lo, hi)
            solved = sum(1 for m in members if m in ever)
            bar_width = 18
            filled = int(round(bar_width * solved / len(members))) if members else 0
            bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
            print(f"    {bar} {solved}/{len(members)}  {tier_name}")
            # Point FORWARD: the next unsolved problem after the one just passed.
            try:
                cur_idx = members.index(num)
            except ValueError:
                cur_idx = -1
            forward = next((m for m in members[cur_idx + 1:] if m not in ever), None)
            earlier_gaps = [m for m in members[:cur_idx] if m not in ever]
            if forward:
                print(f"    next in tier: {forward}")
            elif not earlier_gaps:
                # Everything in this tier is solved — move on.
                if tier_idx + 1 < len(TIERS):
                    _, nlo, nhi = TIERS[tier_idx + 1]
                    nxt = next((m for m in tier_members(nlo, nhi) if m not in ever), None)
                    print(f"    tier complete — next tier starts at {nxt}" if nxt
                          else "    tier complete")
                else:
                    print("    final tier — well done")
            else:
                # Nothing left ahead in this tier, but earlier gaps remain.
                nxt = None
                if tier_idx + 1 < len(TIERS):
                    _, nlo, nhi = TIERS[tier_idx + 1]
                    nxt = next((m for m in tier_members(nlo, nhi) if m not in ever), None)
                if nxt:
                    print(f"    end of tier reached — next: {nxt}")
            if earlier_gaps:
                shown = ", ".join(earlier_gaps[:6]) + ("…" if len(earlier_gaps) > 6 else "")
                print(f"    ({len(earlier_gaps)} earlier still unsolved: {shown})")
        # Count any progress key that looks like a problem ID (digits, optional letter).
        total_solved = sum(1 for k in ever if re.match(r"^\d+[a-z]?$", k))
        print(f"    overall: {total_solved}/{TOTAL_PROBLEMS} solved")
    except Exception:
        pass  # progress display is best-effort


# ---------- machine-readable output (for editor integration) ----------

def _diff_detail(diff):
    """A short human-readable line from a structured diff, so the editor can show
    the same actionable nudge the terminal does (shape/dtype/value + scale)."""
    if not diff:
        return None
    k = diff.get("kind")
    if k == "shape":
        return f"shape: {diff['a']} {diff['shape_a']} vs {diff['b']} {diff['shape_b']}"
    if k == "dtype":
        return f"dtype: {diff['a']} {diff['dtype_a']} vs {diff['b']} {diff['dtype_b']}"
    if k == "value":
        parts = [f"max |{diff['a']} - {diff['b']}| = {diff['max_diff']:.6g}"]
        if diff.get("first_index") is not None:
            parts.append(f"first mismatch at index {tuple(diff['first_index'])}")
        if diff.get("scale"):
            parts.append(f"~{diff['scale']}x scale (check a missing/extra factor)")
        return "; ".join(parts)
    return None


def _progress_payload(num):
    """Compact PASS momentum for editors: (summary string, next-problem id)."""
    try:
        if PREP not in sys.path:
            sys.path.insert(0, PREP)
        from lib.curriculum import find_tier, tier_members, TOTAL_PROBLEMS  # type: ignore
        ever = {k for k, v in _load_progress().items() if v.get("ever_passed")}
        lines, nxt = [], None
        tier = find_tier(num)
        if tier is not None:
            _, tier_name, lo, hi = tier
            members = tier_members(lo, hi)
            solved = sum(1 for m in members if m in ever)
            width = 18
            filled = int(round(width * solved / len(members))) if members else 0
            bar = "[" + "#" * filled + "-" * (width - filled) + "]"
            lines.append(f"{bar} {solved}/{len(members)}  {tier_name}")
            try:
                ci = members.index(num)
            except ValueError:
                ci = -1
            nxt = next((m for m in members[ci + 1:] if m not in ever), None)
        total = sum(1 for k in ever if re.match(r"^\d+[a-z]?$", k))
        lines.append(f"overall: {total}/{TOTAL_PROBLEMS} solved")
        return "\n".join(lines), nxt
    except Exception:
        return None, None


def _result_payload(num, name, stub_path, status, error_type, message, exc):
    """Build the structured run-result dict (shared by the JSON emitter and the
    .last_run.json signal that lets the web UI react to nvim-initiated runs)."""
    out = {
        "status": status,            # "pass" | "fail"
        "problem": num,              # e.g. "05b"
        "name": name,
        "file": os.path.abspath(stub_path),
        "error_type": error_type,    # AssertionError / RuntimeError / NotImplementedError / None
        "message": message,
        "fail_file": None,           # abs path of the user's failing line, if any
        "fail_line": None,
        "fail_func": None,
        "diff": None,                # structured tensor diff, if any
        "detail": None,              # human-readable diff line (FAIL)
        "hint": None,                # one-line likely cause
        "concept": None,             # concept blurb (PASS)
        "progress": None,            # tier bar + overall (PASS)
        "next": None,                # next-problem id (PASS)
    }
    if exc is not None:
        ff, fl, fn_, _line = _user_fail_site(exc)
        out["fail_file"], out["fail_line"], out["fail_func"] = ff, fl, fn_
        out["diff"] = _diff_struct(exc)
        out["detail"] = _diff_detail(out["diff"])
    out["hint"] = _likely_cause(error_type, message, out["diff"])
    if status == "pass":
        try:
            if PREP not in sys.path:
                sys.path.insert(0, PREP)
            from lib.concepts import get_concept  # type: ignore
            out["concept"] = get_concept(num)
        except Exception:
            pass
        out["progress"], out["next"] = _progress_payload(num)
    return out


def _emit_json(num, name, stub_path, status, error_type, message, exc):
    """Print a single JSON object describing the run (PREP_JSON=1)."""
    print(json.dumps(_result_payload(num, name, stub_path, status, error_type, message, exc)))


def _write_last_run(num, name, stub_path, status, error_type, message, exc):
    """Persist the latest run result so the web UI can react to runs started from
    nvim (pp). Written on EVERY run, regardless of PREP_JSON. Best-effort + atomic."""
    out = _result_payload(num, name, stub_path, status, error_type, message, exc)
    out["key"] = f"prob:{num}"
    out["ts"] = time.time()
    try:
        tmp = f"{LAST_RUN_FILE}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, LAST_RUN_FILE)
    except OSError:
        pass


# ---------- main entry ----------

def run_test_for(stub_path):
    base = os.path.basename(stub_path)
    m = re.match(r"^p(\d+[a-z]?)_(.+)\.py$", base)
    if not m:
        print(f"[runner] Not a problem stub: {base}")
        return 2
    num, name = m.groups()
    test_mod_name = f"test_p{num}_{name}"
    compiled = os.path.join(COMPILED_DIR, f"{test_mod_name}.pyc")
    if not os.path.exists(compiled):
        print(f"[runner] Compiled test not found: {compiled}")
        return 2

    if PROBLEMS_DIR not in sys.path:
        sys.path.insert(0, PROBLEMS_DIR)

    loader = SourcelessFileLoader(test_mod_name, compiled)
    spec = importlib.util.spec_from_loader(test_mod_name, loader)
    tmod = importlib.util.module_from_spec(spec)
    label = f"Problem {num} ({name})"
    json_mode = os.environ.get("PREP_JSON") == "1"

    try:
        spec.loader.exec_module(tmod)
        fn = getattr(tmod, test_mod_name)
        fn()
    except NotImplementedError:
        _write_last_run(num, name, stub_path, "fail", "NotImplementedError",
                        "one of the functions still raises NotImplementedError", None)
        if json_mode:
            _emit_json(num, name, stub_path, "fail", "NotImplementedError",
                       "one of the functions still raises NotImplementedError", None)
        else:
            print(f"FAIL {label}: NotImplementedError — one of the functions still raises NotImplementedError.")
        _record(num, False)
        return 1
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message — see test locals below)"
        _write_last_run(num, name, stub_path, "fail", "AssertionError", msg, e)
        if json_mode:
            _emit_json(num, name, stub_path, "fail", "AssertionError", msg, e)
        else:
            print(f"FAIL {label}")
            print(f"  AssertionError: {msg}")
            _print_failure_context(e)
        _record(num, False)
        return 1
    except Exception as e:
        _write_last_run(num, name, stub_path, "fail", type(e).__name__, str(e), e)
        if json_mode:
            _emit_json(num, name, stub_path, "fail", type(e).__name__, str(e), e)
        else:
            print(f"FAIL {label}: {type(e).__name__}: {e}")
            _print_failure_context(e)
        _record(num, False)
        return 1

    # Record BEFORE printing progress, so the dashboard / "next in tier" reflects
    # this pass (otherwise it reads stale .progress.json and appears to go backwards).
    _record(num, True)
    _write_last_run(num, name, stub_path, "pass", None, None, None)
    if json_mode:
        _emit_json(num, name, stub_path, "pass", None, None, None)
    else:
        print(f"PASS {label}")
        _print_concept(num)
        _print_progress_after_pass(num)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python runner.py <path/to/pNN_topic.py>")
        sys.exit(2)
    sys.exit(run_test_for(sys.argv[1]))
