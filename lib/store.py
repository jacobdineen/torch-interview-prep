"""Single SQLite store for all mutable practice state.

Replaces the scattered flat JSON files (.progress.json, .project_progress.json,
.hint_state.json, .solution_unlock.json, .notes.json) with one ACID database, so
concurrent writers (a web `pp` run and an nvim `pp` run happening together) can no
longer lose an update or tear a file. The DB is the source of truth.

For backward compatibility, every write ALSO re-exports the matching JSON file
atomically — readers that still read JSON (web/app.py, check.py --next/--status,
run_all.py, projects.py, and the nvim init.lua that json_decodes .progress.json)
keep working unchanged. SQLite is the truth; the JSON files are a derived cache.

Progress writes merge ever_passed/first_passed in SQL (no read-modify-write race).
WAL + busy_timeout let the short-lived grader subprocesses and the web server
share the DB safely.
"""
import datetime
import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, ".mle_store.db")
ATTEMPTS_CAP = 50000  # keep the run-history table bounded

_SCHEMA = """
CREATE TABLE IF NOT EXISTS problem_progress (
    num TEXT PRIMARY KEY, last_status TEXT, last_run TEXT,
    ever_passed INTEGER DEFAULT 0, first_passed TEXT
);
CREATE TABLE IF NOT EXISTS project_progress (
    project TEXT, step TEXT, last_status TEXT, last_run TEXT,
    ever_passed INTEGER DEFAULT 0, first_passed TEXT,
    PRIMARY KEY (project, step)
);
CREATE TABLE IF NOT EXISTS problem_notes (
    num TEXT, seq INTEGER, text TEXT, PRIMARY KEY (num, seq)
);
CREATE TABLE IF NOT EXISTS project_notes (
    project TEXT, step TEXT, seq INTEGER, text TEXT, PRIMARY KEY (project, step, seq)
);
CREATE TABLE IF NOT EXISTS hint_state (
    num TEXT PRIMARY KEY, count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS problem_unlock (
    num TEXT PRIMARY KEY, unlocked INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS project_unlock (
    project TEXT, step TEXT, unlocked INTEGER DEFAULT 0, PRIMARY KEY (project, step)
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, key TEXT, status TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE INDEX IF NOT EXISTS idx_attempts_key ON attempts(key);
CREATE INDEX IF NOT EXISTS idx_attempts_ts ON attempts(ts);
"""


def _connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    # WAL (set once in init) + synchronous=NORMAL is the standard durable-and-fast
    # combo: commits skip the per-write fsync (only fsync at WAL checkpoint), which
    # is safe across app crashes — only an OS/power crash could lose the last commit,
    # acceptable for progress data. temp_store=MEMORY keeps sorts/temp tables in RAM.
    con.execute("PRAGMA busy_timeout=8000")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    return con


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path, data, sort_keys=True):
    """Write JSON the way the old code did (indent=2, sort_keys) but atomically."""
    try:
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=sort_keys)
        os.replace(tmp, path)
    except OSError:
        pass  # export is best-effort; the DB remains the source of truth


_INIT_DONE = False


def init():
    """Create tables (idempotent) and import any pre-existing JSON exactly once."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    con = _connect()
    try:
        con.execute("PRAGMA journal_mode=WAL")   # persistent (DB header); set once
        con.executescript(_SCHEMA)
        con.commit()
        con.execute("PRAGMA optimize")
        done = con.execute("SELECT v FROM meta WHERE k='migrated'").fetchone()
        if not done:
            _migrate(con)
            con.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('migrated', ?)", (_now(),))
            con.commit()
    finally:
        con.close()
    _INIT_DONE = True


def _migrate(con):
    """One-time import of the legacy flat JSON files into the DB."""
    def load(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    for num, e in load(os.path.join(ROOT, ".progress.json")).items():
        con.execute(
            "INSERT OR REPLACE INTO problem_progress(num,last_status,last_run,ever_passed,first_passed)"
            " VALUES(?,?,?,?,?)",
            (num, e.get("last_status"), e.get("last_run"),
             1 if e.get("ever_passed") else 0, e.get("first_passed")))
    for num, lst in load(os.path.join(ROOT, ".notes.json")).items():
        for i, t in enumerate(lst or []):
            con.execute("INSERT OR REPLACE INTO problem_notes(num,seq,text) VALUES(?,?,?)", (num, i, t))
    for num, c in load(os.path.join(ROOT, ".hint_state.json")).items():
        con.execute("INSERT OR REPLACE INTO hint_state(num,count) VALUES(?,?)", (num, int(c)))
    for num, v in load(os.path.join(ROOT, ".solution_unlock.json")).items():
        con.execute("INSERT OR REPLACE INTO problem_unlock(num,unlocked) VALUES(?,?)", (num, 1 if v else 0))
    proj_root = os.path.join(ROOT, "projects")
    if os.path.isdir(proj_root):
        for name in os.listdir(proj_root):
            d = os.path.join(proj_root, name)
            if not os.path.isdir(d):
                continue
            for step, e in load(os.path.join(d, ".project_progress.json")).items():
                con.execute(
                    "INSERT OR REPLACE INTO project_progress(project,step,last_status,last_run,ever_passed,first_passed)"
                    " VALUES(?,?,?,?,?,?)",
                    (name, step, e.get("last_status"), e.get("last_run"),
                     1 if e.get("ever_passed") else 0, e.get("first_passed")))
            for step, lst in load(os.path.join(d, ".notes.json")).items():
                for i, t in enumerate(lst or []):
                    con.execute("INSERT OR REPLACE INTO project_notes(project,step,seq,text) VALUES(?,?,?,?)",
                                (name, step, i, t))
            for step, v in load(os.path.join(d, ".solution_unlock.json")).items():
                con.execute("INSERT OR REPLACE INTO project_unlock(project,step,unlocked) VALUES(?,?,?)",
                            (name, step, 1 if v else 0))


def _progress_dict(rows):
    """Rebuild the legacy .progress.json entry shape from DB rows."""
    out = {}
    for num, last_status, last_run, ever, first in rows:
        e = {}
        if last_status is not None:
            e["last_status"] = last_status
        if last_run is not None:
            e["last_run"] = last_run
        if ever:
            e["ever_passed"] = True
        if first is not None:
            e["first_passed"] = first
        out[num] = e
    return out


def _export_problem_progress(con):
    rows = con.execute("SELECT num,last_status,last_run,ever_passed,first_passed FROM problem_progress").fetchall()
    _atomic_write_json(os.path.join(ROOT, ".progress.json"), _progress_dict(rows))


def _export_project_progress(con, project, project_dir):
    rows = con.execute(
        "SELECT step,last_status,last_run,ever_passed,first_passed FROM project_progress WHERE project=?",
        (project,)).fetchall()
    _atomic_write_json(os.path.join(project_dir, ".project_progress.json"), _progress_dict(rows))


def _export_problem_notes(con):
    out = {}
    for num, seq, text in con.execute("SELECT num,seq,text FROM problem_notes ORDER BY num,seq"):
        out.setdefault(num, []).append(text)
    _atomic_write_json(os.path.join(ROOT, ".notes.json"), out)


def _export_project_notes(con, project, project_dir):
    out = {}
    for step, seq, text in con.execute(
            "SELECT step,seq,text FROM project_notes WHERE project=? ORDER BY step,seq", (project,)):
        out.setdefault(step, []).append(text)
    _atomic_write_json(os.path.join(project_dir, ".notes.json"), out)


def _export_hint_state(con):
    out = {num: c for num, c in con.execute("SELECT num,count FROM hint_state")}
    _atomic_write_json(os.path.join(ROOT, ".hint_state.json"), out)


def _export_problem_unlock(con):
    out = {num: True for num, u in con.execute("SELECT num,unlocked FROM problem_unlock") if u}
    _atomic_write_json(os.path.join(ROOT, ".solution_unlock.json"), out)


def _export_project_unlock(con, project, project_dir):
    out = {step: True for step, u in
           con.execute("SELECT step,unlocked FROM project_unlock WHERE project=?", (project,)) if u}
    _atomic_write_json(os.path.join(project_dir, ".solution_unlock.json"), out)


def _prune_attempts(con):
    """Keep the history table bounded (cheap no-op while under the cap)."""
    con.execute("DELETE FROM attempts WHERE id <= (SELECT MAX(id) - ? FROM attempts)", (ATTEMPTS_CAP,))


def _project_name(project_dir):
    return os.path.basename(os.path.normpath(project_dir))


def record_problem(num, passed):
    """Atomically record a problem run; ever_passed/first_passed merge in SQL so
    concurrent runs never lose an update. Re-exports .progress.json."""
    init()
    status = "pass" if passed else "fail"
    now = _now()
    con = _connect()
    try:
        con.execute(
            """INSERT INTO problem_progress(num,last_status,last_run,ever_passed,first_passed)
               VALUES(?,?,?,?,?)
               ON CONFLICT(num) DO UPDATE SET
                 last_status=excluded.last_status,
                 last_run=excluded.last_run,
                 ever_passed=MAX(problem_progress.ever_passed, excluded.ever_passed),
                 first_passed=COALESCE(problem_progress.first_passed, excluded.first_passed)""",
            (num, status, now, 1 if passed else 0, now if passed else None))
        con.execute("INSERT INTO attempts(kind,key,status,ts) VALUES('problem',?,?,?)", (num, status, now))
        _prune_attempts(con)
        con.commit()
        _export_problem_progress(con)
    finally:
        con.close()


def load_problem_progress():
    init()
    con = _connect()
    try:
        rows = con.execute(
            "SELECT num,last_status,last_run,ever_passed,first_passed FROM problem_progress").fetchall()
        return _progress_dict(rows)
    finally:
        con.close()


def clear_problems(nums):
    """Delete progress rows for the given problem ids (reset.py)."""
    init()
    con = _connect()
    try:
        con.executemany("DELETE FROM problem_progress WHERE num=?", [(n,) for n in nums])
        con.commit()
        _export_problem_progress(con)
    finally:
        con.close()


def relock_problem(num):
    """Clear progress + hint + unlock for one problem (check.py --redo)."""
    init()
    con = _connect()
    try:
        before = con.total_changes
        con.execute("DELETE FROM problem_progress WHERE num=?", (num,))
        con.execute("DELETE FROM hint_state WHERE num=?", (num,))
        con.execute("DELETE FROM problem_unlock WHERE num=?", (num,))
        changed = con.total_changes > before
        con.commit()
        _export_problem_progress(con)
        _export_hint_state(con)
        _export_problem_unlock(con)
        return changed
    finally:
        con.close()


def relock_project_step(project_dir, sid):
    """Clear progress + solution unlock for one project step (notes are kept)."""
    init()
    project = _project_name(project_dir)
    con = _connect()
    try:
        before = con.total_changes
        con.execute("DELETE FROM project_progress WHERE project=? AND step=?", (project, sid))
        con.execute("DELETE FROM project_unlock WHERE project=? AND step=?", (project, sid))
        changed = con.total_changes > before
        con.commit()
        _export_project_progress(con, project, project_dir)
        _export_project_unlock(con, project, project_dir)
        return changed
    finally:
        con.close()


def reset_project(project_dir):
    """Clear progress + solution unlocks for EVERY step of one project.
    Notes and the attempts history (audit log, surfaced nowhere) are kept."""
    init()
    project = _project_name(project_dir)
    con = _connect()
    try:
        before = con.total_changes
        con.execute("DELETE FROM project_progress WHERE project=?", (project,))
        con.execute("DELETE FROM project_unlock WHERE project=?", (project,))
        cleared = con.total_changes - before
        con.commit()
        _export_project_progress(con, project, project_dir)
        _export_project_unlock(con, project, project_dir)
        return cleared
    finally:
        con.close()


def reset_all_problems():
    """Clear progress + hints + solution unlocks for ALL standalone problems."""
    init()
    con = _connect()
    try:
        before = con.total_changes
        con.execute("DELETE FROM problem_progress")
        con.execute("DELETE FROM hint_state")
        con.execute("DELETE FROM problem_unlock")
        cleared = con.total_changes - before
        con.commit()
        _export_problem_progress(con)
        _export_hint_state(con)
        _export_problem_unlock(con)
        return cleared
    finally:
        con.close()


def reset_everything(project_dirs):
    """Global progress reset: problems + every project in `project_dirs` (each
    project's derived JSON lives in its own dir, hence the explicit list).
    Notes and the attempts history are kept."""
    init()
    con = _connect()
    try:
        before = con.total_changes
        for table in ("problem_progress", "hint_state", "problem_unlock",
                      "project_progress", "project_unlock"):
            con.execute(f"DELETE FROM {table}")  # noqa: S608 — fixed identifier list
        cleared = con.total_changes - before
        con.commit()
        _export_problem_progress(con)
        _export_hint_state(con)
        _export_problem_unlock(con)
        for d in project_dirs:
            project = _project_name(d)
            _export_project_progress(con, project, d)
            _export_project_unlock(con, project, d)
        return cleared
    finally:
        con.close()


def attempts_by_day(days=126):
    """{'YYYY-MM-DD': {'runs': n, 'passes': m}} over the last `days` days, from
    the attempts history (which deliberately survives progress resets). Powers
    the activity heatmap on the web home screen."""
    init()
    con = _connect()
    try:
        rows = con.execute(
            "SELECT substr(ts, 1, 10) AS d, COUNT(*), SUM(status = 'pass')"
            " FROM attempts WHERE d >= date('now', 'localtime', ?) GROUP BY d",
            (f"-{int(days)} day",)).fetchall()
        return {d: {"runs": c, "passes": p or 0} for d, c, p in rows}
    finally:
        con.close()


def record_project(project_dir, step, passed):
    init()
    project = _project_name(project_dir)
    status = "pass" if passed else "fail"
    now = _now()
    con = _connect()
    try:
        con.execute(
            """INSERT INTO project_progress(project,step,last_status,last_run,ever_passed,first_passed)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(project,step) DO UPDATE SET
                 last_status=excluded.last_status,
                 last_run=excluded.last_run,
                 ever_passed=MAX(project_progress.ever_passed, excluded.ever_passed),
                 first_passed=COALESCE(project_progress.first_passed, excluded.first_passed)""",
            (project, step, status, now, 1 if passed else 0, now if passed else None))
        con.execute("INSERT INTO attempts(kind,key,status,ts) VALUES('project',?,?,?)",
                    (f"{project}:{step}", status, now))
        _prune_attempts(con)
        con.commit()
        _export_project_progress(con, project, project_dir)
    finally:
        con.close()


def load_project_progress(project_dir):
    init()
    project = _project_name(project_dir)
    con = _connect()
    try:
        rows = con.execute(
            "SELECT step,last_status,last_run,ever_passed,first_passed FROM project_progress WHERE project=?",
            (project,)).fetchall()
        return _progress_dict(rows)
    finally:
        con.close()


def add_problem_note(num, text):
    init()
    con = _connect()
    try:
        nxt = con.execute("SELECT COALESCE(MAX(seq)+1,0) FROM problem_notes WHERE num=?", (num,)).fetchone()[0]
        con.execute("INSERT INTO problem_notes(num,seq,text) VALUES(?,?,?)", (num, nxt, text))
        con.commit()
        _export_problem_notes(con)
    finally:
        con.close()


def get_problem_notes(num):
    init()
    con = _connect()
    try:
        return [t for (t,) in con.execute("SELECT text FROM problem_notes WHERE num=? ORDER BY seq", (num,))]
    finally:
        con.close()


def add_project_note(project_dir, step, text):
    init()
    project = _project_name(project_dir)
    con = _connect()
    try:
        nxt = con.execute("SELECT COALESCE(MAX(seq)+1,0) FROM project_notes WHERE project=? AND step=?",
                          (project, step)).fetchone()[0]
        con.execute("INSERT INTO project_notes(project,step,seq,text) VALUES(?,?,?,?)",
                    (project, step, nxt, text))
        con.commit()
        _export_project_notes(con, project, project_dir)
    finally:
        con.close()


def get_hint_count(num):
    init()
    con = _connect()
    try:
        r = con.execute("SELECT count FROM hint_state WHERE num=?", (num,)).fetchone()
        return r[0] if r else 0
    finally:
        con.close()


def set_hint_count(num, count):
    init()
    con = _connect()
    try:
        con.execute("INSERT INTO hint_state(num,count) VALUES(?,?) "
                    "ON CONFLICT(num) DO UPDATE SET count=excluded.count", (num, int(count)))
        con.commit()
        _export_hint_state(con)
    finally:
        con.close()


def reset_hint(num):
    init()
    con = _connect()
    try:
        con.execute("DELETE FROM hint_state WHERE num=?", (num,))
        con.commit()
        _export_hint_state(con)
    finally:
        con.close()


def is_problem_unlocked(num):
    init()
    con = _connect()
    try:
        r = con.execute("SELECT unlocked FROM problem_unlock WHERE num=?", (num,)).fetchone()
        return bool(r and r[0])
    finally:
        con.close()


def unlock_problem(num):
    init()
    con = _connect()
    try:
        con.execute("INSERT INTO problem_unlock(num,unlocked) VALUES(?,1) "
                    "ON CONFLICT(num) DO UPDATE SET unlocked=1", (num,))
        con.commit()
        _export_problem_unlock(con)
    finally:
        con.close()


def is_project_unlocked(project_dir, step):
    init()
    con = _connect()
    try:
        r = con.execute("SELECT unlocked FROM project_unlock WHERE project=? AND step=?",
                        (_project_name(project_dir), step)).fetchone()
        return bool(r and r[0])
    finally:
        con.close()


def unlock_project(project_dir, step):
    init()
    project = _project_name(project_dir)
    con = _connect()
    try:
        con.execute("INSERT INTO project_unlock(project,step,unlocked) VALUES(?,?,1) "
                    "ON CONFLICT(project,step) DO UPDATE SET unlocked=1", (project, step))
        con.commit()
        _export_project_unlock(con, project, project_dir)
    finally:
        con.close()
