"""Project CLI — multi-step projects (e.g. tiny-gpt-from-scratch).

  python projects.py                       # list projects / the current one
  python projects.py <project>             # list its parts + steps ([x]/[ ])
  python projects.py <project> --status    # per-part progress bars
  python projects.py <project> --next      # next unsolved step (id + path)
  python projects.py <project> <id>        # run that step
  python projects.py <project> <id> --explain   # signature + what it does + notes
  python projects.py <project> <id> --solution [--i-give-up]   # reference (unlock persists)
  python projects.py <project> <id> --note "TEXT"   # jot a note; shown under --explain
  python projects.py <project> --scaffold  # run the end-to-end demo

<project> may be omitted when there is exactly one project. <id> is a 4-digit
step id ('0044') or its number ('44').
"""
import argparse
import ast
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(HERE, "projects")


def _all_projects():
    return sorted(d for d in glob.glob(os.path.join(PROJECTS_DIR, "*"))
                  if os.path.exists(os.path.join(d, "project.json")))


def _resolve_project(name):
    projs = _all_projects()
    if name:
        hit = [p for p in projs if os.path.basename(p) == name]
        return hit[0] if hit else None
    return projs[0] if len(projs) == 1 else None


def _manifest(root):
    with open(os.path.join(root, "project.json")) as f:
        return json.load(f)


def _progress(root):
    pf = os.path.join(root, ".project_progress.json")
    if not os.path.exists(pf):
        return {}
    try:
        with open(pf) as f:
            return json.load(f)
    except Exception:
        return {}


def _norm_id(arg):
    s = arg.lstrip("0") or "0"
    if s.isdigit():
        return f"{int(s):04d}"
    return arg


def _step_path(root, step):
    return os.path.join(root, "steps", f"{step['id']}_{step['name']}.py")


def _bar(done, total, w=18):
    f = int(round(w * done / total)) if total else 0
    return "[" + "#" * f + "-" * (w - f) + "]"


def cmd_list(root):
    man, prog = _manifest(root), _progress(root)
    ever = {k for k, v in prog.items() if v.get("ever_passed")}
    print(f"\n  {man['title']}  ({sum(1 for s in man['steps'] if s['id'] in ever)}/{len(man['steps'])} steps)\n")
    for pi, part in enumerate(man["parts"]):
        members = [s for s in man["steps"] if s["part"] == pi]
        done = sum(1 for s in members if s["id"] in ever)
        print(f"  Part {pi + 1}: {part['title']}  {_bar(done, len(members))} {done}/{len(members)}")
        for s in members:
            mark = "x" if s["id"] in ever else " "
            built = "" if os.path.exists(_step_path(root, s)) else "  (not built)"
            print(f"     [{mark}] {s['id']}  {s['name']}{built}")
        print()


def cmd_overview(projs):
    """List every project with a one-line progress summary (no project named)."""
    print(f"\n  Projects ({len(projs)}):\n")
    for root in projs:
        man, prog = _manifest(root), _progress(root)
        ever = {k for k, v in prog.items() if v.get("ever_passed")}
        total = len(man["steps"])
        done = sum(1 for s in man["steps"] if s["id"] in ever)
        name = os.path.basename(root)
        print(f"  {_bar(done, total)} {done:>3}/{total:<3}  {name:<22}  {man['title']}")
    print("\n  python projects.py <name>          # parts + steps, [x]/[ ] solved")
    print("  python projects.py <name> --next   # jump to the next unsolved step\n")


def cmd_status(root):
    man, prog = _manifest(root), _progress(root)
    ever = {k for k, v in prog.items() if v.get("ever_passed")}
    print()
    total_done = 0
    for pi, part in enumerate(man["parts"]):
        members = [s for s in man["steps"] if s["part"] == pi]
        done = sum(1 for s in members if s["id"] in ever)
        total_done += done
        nxt = next((s["id"] for s in members if s["id"] not in ever), None)
        tail = f"  next: {nxt}" if nxt else "  (complete)"
        print(f"  {_bar(done, len(members))} {done:>2}/{len(members):<2}  Part {pi + 1}: {part['title']:<34}{tail}")
    print(f"\n  Overall: {total_done}/{len(man['steps'])} steps\n")


def cmd_next(root, after=None):
    man, prog = _manifest(root), _progress(root)
    ever = {k for k, v in prog.items() if v.get("ever_passed")}
    steps = [s for s in man["steps"] if os.path.exists(_step_path(root, s))]
    ids = [s["id"] for s in steps]
    by_id = {s["id"]: s for s in steps}
    chosen, wrapped = None, False
    if after and _norm_id(after) in ids:
        i = ids.index(_norm_id(after))
        chosen = next((x for x in ids[i + 1:] if x not in ever), None)
    if chosen is None:
        # Nothing ahead of `after`; fall back to the earliest gap and say so, so
        # the learner knows they're backfilling rather than advancing.
        chosen = next((x for x in ids if x not in ever), None)
        wrapped = bool(after) and chosen is not None
    if chosen is None:
        print("All built steps solved — run --scaffold.")
        return
    if wrapped:
        print(f"Nothing left ahead — earlier gap: {chosen}  {by_id[chosen]['name']}")
    else:
        print(f"Next unsolved: {chosen}  {by_id[chosen]['name']}")
    print(_step_path(root, by_id[chosen]))


def _find_step(man, sid):
    return next((s for s in man["steps"] if s["id"] == sid), None)


# ---- per-project notes + persistent solution unlock (parity with the problems side) ----

def _json_store(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _notes_file(root):
    return os.path.join(root, ".notes.json")


def _unlock_file(root):
    return os.path.join(root, ".solution_unlock.json")


def cmd_note(root, sid, text):
    man = _manifest(root)
    if not _find_step(man, sid):
        print(f"no step {sid}")
        return
    path = _notes_file(root)
    notes = _json_store(path)
    notes.setdefault(sid, []).append(text)
    with open(path, "w") as f:
        json.dump(notes, f, indent=2, sort_keys=True)
    print(f"  noted for {sid} (shown under --explain).")


def cmd_explain(root, sid):
    man = _manifest(root)
    s = _find_step(man, sid)
    if not s:
        print(f"no step {sid}")
        return
    part = man["parts"][s["part"]]
    print()
    print(f"  Step {s['id']} — {s['name']}   (Part {s['part'] + 1}: {part['title']})")
    print(f"    signature: {s['signature']}")
    print(f"    {s['doc']}")
    if part.get("description"):
        print(f"    Part: {part['description']}")
    notes = _json_store(_notes_file(root)).get(sid)
    if notes:
        print("    Your notes:")
        for n in notes:
            print(f"      - {n}")


def _reference_source(root, name):
    ref = os.path.join(root, "tests", "_ref", "reference.py")
    with open(ref) as f:
        src = f.read()
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in tree.body:
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name):
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    return None


def cmd_solution(root, sid, i_give_up):
    man, prog = _manifest(root), _progress(root)
    s = _find_step(man, sid)
    if not s:
        print(f"no step {sid}")
        return
    unlocks = _json_store(_unlock_file(root))
    if i_give_up and not unlocks.get(sid):
        unlocks[sid] = True
        with open(_unlock_file(root), "w") as f:
            json.dump(unlocks, f, indent=2, sort_keys=True)
    ok = prog.get(sid, {}).get("ever_passed") or i_give_up or unlocks.get(sid)
    if not ok:
        print(f"\n  Solution for {sid} is locked. Pass it once, or re-run with --i-give-up.")
        return
    src = _reference_source(root, s["name"])
    print(f"\n  Reference for {sid} ({s['name']}):\n")
    for ln in (src or "  (not found)").split("\n"):
        print(f"    {ln}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("project", nargs="?")
    p.add_argument("id", nargs="?")
    p.add_argument("--status", action="store_true")
    p.add_argument("--next", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--explain", action="store_true")
    p.add_argument("--solution", action="store_true")
    p.add_argument("--i-give-up", action="store_true")
    p.add_argument("--note", metavar="TEXT",
                   help="save a free-text note for this step (shown later under --explain)")
    p.add_argument("--scaffold", action="store_true")
    args = p.parse_args()

    # If the first positional looks like a step id and there's one project, shift.
    projs = _all_projects()
    if args.project and args.id is None and (args.project.isdigit() or len(args.project) == 4):
        if len(projs) == 1:
            args.id, args.project = args.project, None

    if not projs:
        print("No projects found under projects/.")
        return
    root = _resolve_project(args.project)
    if root is None:
        if args.project:
            print(f"No project named '{args.project}'. Available: "
                  + ", ".join(os.path.basename(x) for x in projs))
            sys.exit(2)
        # No project named and more than one exists: show the overview.
        return cmd_overview(projs)

    if args.scaffold:
        sys.exit(subprocess.run([sys.executable, os.path.join(root, "scaffold.py")]).returncode)
    if args.next:
        return cmd_next(root, after=args.id)
    if args.status:
        return cmd_status(root)
    if args.id is None or args.list:
        return cmd_list(root)

    sid = _norm_id(args.id)
    if args.note is not None:
        return cmd_note(root, sid, args.note)
    if args.explain:
        return cmd_explain(root, sid)
    if args.solution:
        return cmd_solution(root, sid, args.i_give_up)

    # default: run the step
    man = _manifest(root)
    s = _find_step(man, sid)
    if not s:
        print(f"no step {sid}")
        sys.exit(2)
    path = _step_path(root, s)
    if not os.path.exists(path):
        print(f"step {sid} ({s['name']}) not built yet")
        sys.exit(2)
    os.execv(sys.executable, [sys.executable, path])


if __name__ == "__main__":
    main()
