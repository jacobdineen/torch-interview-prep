"""Regenerate every project from its spec: run each projects/*/_build/gen.py.

  python regen_all.py        # regen all projects (stubs + project.json + compiled tests)

After editing a project's spec.py (or shared build logic), this rebuilds every
project so no project.json is left stale. gen.py is idempotent and never
clobbers in-progress step work. Exits non-zero if any project fails to build.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def main():
    gens = sorted(glob.glob(os.path.join(HERE, "projects", "*", "_build", "gen.py")))
    if not gens:
        print("No projects found under projects/.")
        return 0
    failed = []
    for gen in gens:
        name = os.path.basename(os.path.dirname(os.path.dirname(gen)))
        print(f"\n=== {name} ===")
        rc = subprocess.run([sys.executable, gen], cwd=os.path.dirname(os.path.dirname(gen))).returncode
        if rc != 0:
            failed.append(name)
    print(f"\nregenerated {len(gens) - len(failed)}/{len(gens)} projects")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
