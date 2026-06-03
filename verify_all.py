"""Run every correctness check: the problem reference solutions and each project's
reference. Used by CI (and handy locally) to ensure a change hasn't broken any
problem or project.

  python verify_all.py        # exit 0 only if everything passes
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(label, cmd):
    print(f"\n{'=' * 60}\n# {label}\n{'=' * 60}")
    rc = subprocess.run(cmd, cwd=HERE).returncode
    print(f"--> {label}: {'PASS' if rc == 0 else 'FAIL'} (exit {rc})")
    return rc


def main():
    py = sys.executable
    results = {}
    results["framework"] = _run("framework (grader + assembler)",
                                 [py, os.path.join(HERE, "test_framework.py")])
    results["problems"] = _run("problems (reference solutions)",
                               [py, os.path.join(HERE, "verify_problems.py")])
    results["numpy"] = _run("numpy variants (via the torch<->numpy bridge)",
                            [py, os.path.join(HERE, "verify_numpy.py")])
    for verify in sorted(glob.glob(os.path.join(HERE, "projects", "*", "_build", "verify.py"))):
        name = os.path.basename(os.path.dirname(os.path.dirname(verify)))
        results[name] = _run(f"project: {name}", [py, verify])

    print(f"\n{'=' * 60}\n# summary\n{'=' * 60}")
    for k, rc in results.items():
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {k}")
    failed = [k for k, rc in results.items() if rc != 0]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
