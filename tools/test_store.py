"""Unit tests for lib/store.py against an ISOLATED temp database.

Covers the contracts the app depends on: atomic pass/fail recording with
monotonic ever_passed, per-scope resets (item / project / problems / all),
notes + attempts surviving resets, hint/unlock state, the JSON re-exports,
attempts_by_day, and concurrent recording. Run by verify_all.py / CI.

  python tools/test_store.py
"""
import datetime
import json
import os
import shutil
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from lib import store  # noqa: E402

PASSED = 0


def check(cond, label):
    global PASSED
    assert cond, f"FAIL: {label}"
    PASSED += 1
    print(f"  ok  {label}")


def main():
    tmp = tempfile.mkdtemp(prefix="mle_store_test_")
    proj_a = os.path.join(tmp, "projects", "proj-a")
    proj_b = os.path.join(tmp, "projects", "proj-b")
    os.makedirs(proj_a)
    os.makedirs(proj_b)
    # Point the store at the sandbox (module-level anchors, set before init()).
    store.ROOT = tmp
    store.DB_PATH = os.path.join(tmp, ".mle_store.db")
    store._INIT_DONE = False

    try:
        # --- recording + monotonic ever_passed
        store.record_problem("01a", False)
        p = store.load_problem_progress()
        check(p["01a"]["last_status"] == "fail" and "ever_passed" not in p["01a"],
              "fail recorded, not ever_passed")
        store.record_problem("01a", True)
        store.record_problem("01a", False)
        p = store.load_problem_progress()
        check(p["01a"]["ever_passed"] and p["01a"]["last_status"] == "fail",
              "ever_passed survives a later fail (monotonic)")
        first = p["01a"]["first_passed"]
        store.record_problem("01a", True)
        check(store.load_problem_progress()["01a"]["first_passed"] == first,
              "first_passed never moves")

        # --- JSON re-export mirrors the DB
        exported = json.load(open(os.path.join(tmp, ".progress.json")))
        check(exported == store.load_problem_progress(), ".progress.json mirrors the DB")

        # --- notes / hints / unlocks
        store.add_problem_note("01a", "note 1")
        store.set_hint_count("01a", 2)
        store.unlock_problem("01a")
        check(store.get_problem_notes("01a") == ["note 1"], "notes round-trip")
        check(store.get_hint_count("01a") == 2 and store.is_problem_unlocked("01a"),
              "hint count + unlock round-trip")

        # --- relock (per-item reset) clears progress/hint/unlock, keeps notes
        check(store.relock_problem("01a") is True, "relock reports a change")
        check(store.load_problem_progress() == {} and store.get_hint_count("01a") == 0
              and not store.is_problem_unlocked("01a"), "relock cleared progress/hint/unlock")
        check(store.get_problem_notes("01a") == ["note 1"], "relock keeps notes")

        # --- project scopes are isolated
        store.record_project(proj_a, "0001", True)
        store.record_project(proj_b, "0001", True)
        store.unlock_project(proj_a, "0001")
        store.add_project_note(proj_a, "0001", "keep")
        check(store.relock_project_step(proj_a, "0001") is True, "step relock")
        check(store.load_project_progress(proj_a) == {}
              and store.load_project_progress(proj_b) != {},
              "step relock leaves the other project alone")
        store.record_project(proj_a, "0001", True)
        store.record_project(proj_a, "0002", False)
        check(store.reset_project(proj_a) >= 2, "project reset clears its rows")
        check(store.load_project_progress(proj_b) != {}, "project reset is scoped")
        check(json.load(open(os.path.join(proj_a, ".project_progress.json"))) == {},
              "project reset re-exports its JSON")

        # --- problems-only and global resets
        store.record_problem("02a", True)
        check(store.reset_all_problems() >= 1, "problems reset")
        check(store.load_project_progress(proj_b) != {}, "problems reset spares projects")
        store.record_problem("03a", True)
        check(store.reset_everything([proj_a, proj_b]) >= 1, "global reset")
        check(store.load_problem_progress() == {} == store.load_project_progress(proj_b),
              "global reset clears everything")
        notes_after = store.get_problem_notes("01a")
        check(notes_after == ["note 1"], "global reset keeps notes")

        # --- attempts history survives resets and powers the heatmap
        today = datetime.date.today().isoformat()
        days = store.attempts_by_day()
        check(today in days and days[today]["runs"] >= 8 and days[today]["passes"] >= 5,
              "attempts_by_day aggregates today's history (post-reset)")

        # --- concurrent recording never loses ever_passed
        def hammer(i):
            for j in range(25):
                store.record_problem("99z", (i + j) % 2 == 0)
        ts = [threading.Thread(target=hammer, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        check(store.load_problem_progress()["99z"]["ever_passed"],
              "ever_passed survives 100 concurrent mixed records")

        print(f"\nALL {PASSED} store checks passed.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
