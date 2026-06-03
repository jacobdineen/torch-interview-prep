"""One dashboard for the whole curriculum: where am I, and what's next.

  python prep.py        # problem tiers + every project's progress + the next action

A single "where am I / what do I do now" view so the daily restart doesn't mean
stitching together run_all.py --status and projects.py --status per project.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    # Problems: per-tier bars (the standalone problems are the PyTorch track).
    from run_all import show_status
    print("\n=== Problems  (PyTorch track) ===")
    show_status()

    # Projects: grouped by framework (numpy vs torch) so the two tracks separate.
    from projects import _all_projects, cmd_overview
    projs = _all_projects()
    if projs:
        print("=== Projects  (by framework) ===")
        cmd_overview(projs)

    # The single next action for the problems track.
    from check import _next_unsolved
    pid, path, wrapped = _next_unsolved()
    print("=== Next ===")
    if not pid:
        print("  All problems solved — pick a project above, or drill with `check.py <id> --redo`.\n")
        return
    lead = "earlier gap" if wrapped else "next problem"
    print(f"  {lead}: {pid}")
    if path:
        print(f"    {path}")
    print(f"\n  Resume:  uv run python check.py            # jump straight to {pid}\n")


if __name__ == "__main__":
    main()
