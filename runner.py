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
import traceback
from importlib.machinery import SourcelessFileLoader

PREP = os.path.dirname(os.path.abspath(__file__))
PROBLEMS_DIR = os.path.join(PREP, "problems")
COMPILED_DIR = os.path.join(PREP, "tests", "_compiled")
PROGRESS_FILE = os.path.join(PREP, ".progress.json")

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


# ---------- progress tracking ----------

def _load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_progress(progress):
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2, sort_keys=True)
    except Exception:
        pass  # progress tracking is best-effort; never block a run on it


def _record(num, passed):
    progress = _load_progress()
    entry = progress.get(num, {})
    entry["last_status"] = "pass" if passed else "fail"
    entry["last_run"] = datetime.datetime.now().isoformat(timespec="seconds")
    if passed:
        entry["ever_passed"] = True
        entry["first_passed"] = entry.get("first_passed") or entry["last_run"]
    progress[num] = entry
    _save_progress(progress)


# ---------- concept blurbs ----------

def _print_concept(num):
    try:
        if PREP not in sys.path:
            sys.path.insert(0, PREP)
        from concepts import get_concept  # type: ignore
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
        from curriculum import find_tier, tier_members, TIERS, TOTAL_PROBLEMS  # type: ignore
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
            next_in_tier = next((m for m in members if m not in ever), None)
            if next_in_tier:
                print(f"    next in tier: {next_in_tier}")
            elif tier_idx + 1 < len(TIERS):
                _, nlo, nhi = TIERS[tier_idx + 1]
                next_problem = next((m for m in tier_members(nlo, nhi) if m not in ever), None)
                if next_problem:
                    print(f"    tier complete — next tier starts at {next_problem}")
                else:
                    print("    tier complete")
            else:
                print("    final tier — well done")
        # Count any progress key that looks like a problem ID (digits, optional letter).
        total_solved = sum(1 for k in ever if re.match(r"^\d+[a-z]?$", k))
        print(f"    overall: {total_solved}/{TOTAL_PROBLEMS} solved")
    except Exception:
        pass  # progress display is best-effort


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

    try:
        spec.loader.exec_module(tmod)
        fn = getattr(tmod, test_mod_name)
        fn()
    except NotImplementedError:
        print(f"FAIL {label}: NotImplementedError — one of the functions still raises NotImplementedError.")
        _record(num, False)
        return 1
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message — see test locals below)"
        print(f"FAIL {label}")
        print(f"  AssertionError: {msg}")
        _print_failure_context(e)
        _record(num, False)
        return 1
    except Exception as e:
        print(f"FAIL {label}: {type(e).__name__}: {e}")
        _print_failure_context(e)
        _record(num, False)
        return 1

    print(f"PASS {label}")
    _record(num, True)
    _print_concept(num)
    _print_progress_after_pass(num)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python runner.py <path/to/pNN_topic.py>")
        sys.exit(2)
    sys.exit(run_test_for(sys.argv[1]))
