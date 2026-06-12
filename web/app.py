"""Web-app backend for the practice curriculum (stdlib only).

A thin HTTP API + static server that pairs a problem-description UI with your
REAL Neovim as the editor (embedded via ttyd in the page). The editor isn't an
emulation: the API drives one persistent nvim over its --listen socket, so
opening an item is `nvim --remote <file>` and Run is a remote `:wa` + the test.

The catalog spans BOTH standalone problems and every project's steps. Items are
addressed by a key: `prob:<id>` (e.g. prob:04a) or `proj:<name>:<sid>`
(e.g. proj:tiny-gpt-from-scratch:0012).

  NVIM_SOCK=/tmp/mle_nvim.sock TTYD_PORT=7681 APIPORT=8000 python web/app.py

Started together with ttyd by web/serve-app.sh.
"""
import ast
import glob
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

def _load_secret():
    """Persist the CSRF token across restarts so an already-open tab keeps working
    (a fresh token each start would 403 every open POST until the user refreshes)."""
    path = "/tmp/mle_prep_token"
    try:
        t = open(path).read().strip()
        if len(t) >= 16:
            return t
    except Exception:
        pass
    t = secrets.token_hex(16)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.write(fd, t.encode()); os.close(fd)
    except Exception:
        pass
    return t


_SECRET = _load_secret()                # CSRF token for mutating POSTs (exposed via same-origin /api/config)
_run_lock = threading.Lock()           # serialize /api/run (avoid racing check.py + torn progress files)
_catalog_cache = {"sig": None, "data": None}

WEB = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WEB)
LAST_RUN = os.path.join(ROOT, ".last_run.json")  # latest run result (any front-end) — drives /api/sync
STATIC = os.path.join(WEB, "static")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NVIM_SOCK = os.environ.get("NVIM_SOCK", "/tmp/mle_nvim.sock")
TTYD_PORT = int(os.environ.get("TTYD_PORT", "7681"))
APIPORT = int(os.environ.get("APIPORT", "8000"))
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable


# ---------- shared ----------

def _signature(tree):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = ", ".join(a.arg for a in node.args.args)
            return f"def {node.name}({args})"
        if isinstance(node, ast.ClassDef):
            return f"class {node.name}"
    return ""


def _load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _strip_example(text):
    """Drop a trailing "Example:" block from a docstring body. The web renders
    the worked example in its own box, so showing it in the prose too is just
    noise (and risks reading like part of the spec)."""
    out = []
    for ln in text.splitlines():
        if ln.strip() == "Example:":
            break
        out.append(ln)
    return "\n".join(out).strip()


# ---------- problems ----------

def _problem_path(pid):
    hits = sorted(glob.glob(os.path.join(ROOT, "problems", f"p{pid}_*.py")))
    return hits[0] if hits else None


def _problem_progress():
    return _load_json(os.path.join(ROOT, ".progress.json"))


def _problem_title(pid):
    path = _problem_path(pid)
    if not path:
        return pid
    try:
        with open(path) as f:
            doc = (ast.get_docstring(ast.parse(f.read())) or "").splitlines()
        if doc:
            m = re.match(r"Problem\s+\S+:\s*(.+)", doc[0])
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    return pid


def _tier(pid):
    try:
        from lib.curriculum import find_tier
        t = find_tier(pid)
        return t[1] if t else ""
    except Exception:
        return ""


# ---------- projects ----------

def _project_dirs():
    return sorted(d for d in glob.glob(os.path.join(ROOT, "projects", "*"))
                  if os.path.isfile(os.path.join(d, "project.json")))


def _manifest(proj_dir):
    return _load_json(os.path.join(proj_dir, "project.json"))


def _project_progress(proj_dir):
    return _load_json(os.path.join(proj_dir, ".project_progress.json"))


def _step_path(proj_dir, sid, name):
    return os.path.join(proj_dir, "steps", f"{sid}_{name}.py")


def _part(man, idx):
    parts = man.get("parts", [])
    return parts[idx] if 0 <= idx < len(parts) else {"title": "", "description": ""}


# ---------- catalog + resolution ----------

def _catalog_sig():
    paths = glob.glob(os.path.join(ROOT, "problems", "p*_*.py")) + \
            glob.glob(os.path.join(ROOT, "projects", "*", "project.json"))
    try:
        return (len(paths), max((os.path.getmtime(p) for p in paths), default=0))
    except OSError:
        return None


def _catalog():
    """Cached: rebuild only when a problem/project file changes (mtime)."""
    sig = _catalog_sig()
    if sig is not None and _catalog_cache["sig"] == sig and _catalog_cache["data"] is not None:
        return _catalog_cache["data"]
    data = _build_catalog()
    _catalog_cache["sig"], _catalog_cache["data"] = sig, data
    return data


def _build_catalog():
    """All selectable items (problems + built project steps), with a source list
    for grouping/filtering in the picker."""
    items, sources = [], ["Problems"]
    prog = _problem_progress()
    try:
        from lib.curriculum import all_problem_ids
        ids = all_problem_ids()
    except Exception:
        ids = []
    from lib import frameworks
    for pid in ids:
        items.append({"key": f"prob:{pid}", "kind": "problem", "source": "Problems",
                      "group": _tier(pid) or "Problems", "id": pid,
                      "title": _problem_title(pid), "framework": frameworks.problem_framework(pid),
                      "numpy": frameworks.problem_has_numpy(pid),
                      "solved": bool(prog.get(pid, {}).get("ever_passed")),
                      "last_status": prog.get(pid, {}).get("last_status")})
    projects = {}
    for d in _project_dirs():
        man = _manifest(d)
        name = man.get("name", os.path.basename(d))
        sources.append(name)
        parts = man.get("parts", [])
        projects[name] = {"title": man.get("title", name),
                          "description": (parts[0].get("description") if parts else "") or ""}
        fw = frameworks.project_framework(d)
        pp = _project_progress(d)
        for s in man.get("steps", []):
            if not os.path.exists(_step_path(d, s["id"], s["name"])):
                continue  # staged build: not yet generated
            items.append({"key": f"proj:{name}:{s['id']}", "kind": "project",
                          "source": name, "framework": fw,
                          "group": f"Part {s.get('part', 0) + 1}: {_part(man, s.get('part', 0))['title']}",
                          "id": s["id"], "title": s["name"],
                          "solved": bool(pp.get(s["id"], {}).get("ever_passed")),
                          "last_status": pp.get(s["id"], {}).get("last_status")})
    return {"sources": sources, "items": items, "projects": projects,
            "frameworks": [frameworks.NUMPY, frameworks.TORCH]}


def _resolve(key):
    if key.startswith("prob:"):
        pid = key[5:]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pid):
            return None
        path = _problem_path(pid)
        return {"kind": "problem", "pid": pid, "path": path} if path else None
    if key.startswith("proj:"):
        try:
            _, name, sid = key.split(":", 2)
        except ValueError:
            return None
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not re.fullmatch(r"[A-Za-z0-9_]+", sid):
            return None
        d = os.path.join(ROOT, "projects", name)
        if not os.path.isfile(os.path.join(d, "project.json")):
            return None
        man = _manifest(d)
        s = next((x for x in man.get("steps", []) if x["id"] == sid), None)
        if not s:
            return None
        path = _step_path(d, sid, s["name"])
        if not os.path.exists(path):
            return None
        return {"kind": "project", "name": name, "sid": sid, "step": s,
                "manifest": man, "dir": d, "path": path}
    return None


def _item_meta(key):
    r = _resolve(key)
    if not r:
        return None
    if r["kind"] == "problem":
        pid = r["pid"]
        with open(r["path"]) as f:
            src = f.read()
        try:
            tree = ast.parse(src)
            full = (ast.get_docstring(tree) or "").strip()
        except SyntaxError:
            tree, full = None, ""
        # Body = docstring minus the "Problem NN: title" line (shown as the H1)
        # and the internal "(Split from parent ...)" artifact.
        body = [ln for ln in full.splitlines()[1:]
                if not ln.strip().startswith("(Split from parent problem")]
        prog = _problem_progress().get(pid, {})
        concept = ""
        try:
            from lib.concepts import get_concept
            concept = get_concept(pid) or ""
        except Exception:
            pass
        from lib import frameworks
        slug = os.path.basename(r["path"])[len(f"p{pid}_"):-3]
        example = None
        try:
            from lib.examples import example_for
            example = example_for(pid, slug)
        except Exception:
            pass
        return {"key": key, "kind": "problem", "id": pid, "title": _problem_title(pid),
                "source": "Problems", "group": _tier(pid),
                "framework": frameworks.framework_of_source(src),
                "numpy": frameworks.problem_has_numpy(pid),
                "signature": _signature(tree) if tree else "",
                "doc": _strip_example("\n".join(body)), "concept": concept, "example": example,
                "solved": bool(prog.get("ever_passed")), "last_status": prog.get("last_status")}
    from lib import frameworks
    man, s = r["manifest"], r["step"]
    part = _part(man, s.get("part", 0))
    pp = _project_progress(r["dir"]).get(s["id"], {})
    doc = _strip_example(s.get("doc", ""))
    if part.get("description"):
        doc = (doc + "\n\n" + part["description"]).strip()
    example = None
    try:
        from lib.examples import example_for_step
        example = example_for_step(r["name"], s["id"], s["name"])
    except Exception:
        pass
    return {"key": key, "kind": "project", "id": s["id"], "title": s["name"],
            "source": man.get("title", r["name"]),
            "group": f"Part {s.get('part', 0) + 1}: {part['title']}",
            "framework": frameworks.project_framework(r["dir"]),
            "signature": s.get("signature", ""), "doc": doc, "concept": "",
            "example": example,
            "solved": bool(pp.get("ever_passed")), "last_status": pp.get("last_status")}


# ---------- nvim remote control ----------

def _nvim(*args, timeout=10):
    try:
        return subprocess.run(["nvim", "--server", NVIM_SOCK, *args],
                              capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return subprocess.CompletedProcess(args, 1, "", str(e))


def nvim_open(path):
    # Short timeout: a busy/stale nvim must not block the open request for long.
    return _nvim("--remote", path, timeout=3).returncode == 0


def _ensure_server():
    """Revive the persistent nvim server if it died (e.g. :qa! or a crash) while a
    browser tab stayed connected, so Run doesn't fail until a manual reconnect."""
    try:
        subprocess.run(["bash", os.path.join(WEB, "mle-nvim-ensure.sh"), NVIM_SOCK], timeout=15)
    except Exception:
        pass


def _save_once(path):
    nvim_open(path)   # focus the graded buffer so we save the RIGHT file
    # --remote-expr BLOCKS until nvim evaluates it (no fire-and-forget race where
    # check.py reads pre-save content) and writes ONLY the current buffer (never :wa).
    r = _nvim("--remote-expr", "execute('silent! update') . '|ok'", timeout=6)
    return r.returncode == 0 and (r.stdout or "").strip().endswith("|ok")


def nvim_save(path):
    """Synchronously write the graded file. If the server is dead, revive once + retry."""
    if _save_once(path):
        return True
    _ensure_server()
    return _save_once(path)


def teardown():
    """Stop everything: save + quit the editor, stop ttyd, drop the socket, then
    exit this API process. Idempotent and best-effort."""
    try:
        _nvim("--remote-send", "<C-\\><C-N>:silent! w<CR>:qa!<CR>", timeout=3)
    except Exception:
        pass
    pid = os.environ.get("TTYD_PID")
    try:
        if pid:
            os.kill(int(pid), signal.SIGTERM)
        else:
            subprocess.run(["pkill", "-f", f"ttyd.*-p {TTYD_PORT}"], timeout=5)
    except Exception:
        pass
    try:
        with open("/tmp/mle_nvim_server.pid") as f:
            os.kill(int(f.read().strip()), signal.SIGTERM)
    except Exception:
        pass
    try:
        if os.path.exists(NVIM_SOCK):
            os.remove(NVIM_SOCK)
        if os.path.exists("/tmp/mle_nvim_server.pid"):
            os.remove("/tmp/mle_nvim_server.pid")
    except Exception:
        pass
    threading.Timer(0.4, lambda: os._exit(0)).start()


# ---------- run / debug ----------

def _parse_json_line(proc):
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue
    msg = (proc.stderr or proc.stdout or "no output").strip()[:4000]
    rc = getattr(proc, "returncode", 0)
    if rc not in (0, 1):
        msg = f"(exit {rc}) {msg}"
    return {"status": "error", "message": msg}


def _path_to_key(path):
    """Map an absolute problem/step file path back to a catalog key (reverse of _resolve)."""
    if not path:
        return None
    path = os.path.abspath(path)
    m = re.search(r"/problems/p([A-Za-z0-9]+)_[^/]+\.py$", path)
    if m:
        return f"prob:{m.group(1)}"
    m = re.search(r"/projects/([^/]+)/steps/(\d+)_[^/]+\.py$", path)
    if m:
        return f"proj:{m.group(1)}:{m.group(2)}"
    return None


_lastrun_cache = {"mtime": None, "data": None}
_curkey_cache = {"ts": 0.0, "key": None}


def _read_last_run():
    """Parse .last_run.json only when it actually changed (mtime-gated) — the poll
    hits this every ~1.2s, so avoid re-parsing on every tick."""
    try:
        mt = os.path.getmtime(LAST_RUN)
    except OSError:
        return None
    if _lastrun_cache["mtime"] != mt:
        try:
            with open(LAST_RUN) as f:
                _lastrun_cache["data"] = json.load(f)
        except Exception:
            _lastrun_cache["data"] = None
        _lastrun_cache["mtime"] = mt
    return _lastrun_cache["data"]


def _current_nvim_key():
    """Which problem/step the editor is showing (so the page can follow nvim nav).
    The probe forks an nvim client, so skip it while a run holds the lock (nvim is
    busy then) and cache briefly so the poll doesn't fork nvim on every tick."""
    now = time.monotonic()
    if _run_lock.locked():
        return _curkey_cache["key"]
    if now - _curkey_cache["ts"] < 2.0:
        return _curkey_cache["key"]
    r = _nvim("--remote-expr", "expand('%:p')", timeout=2)
    key = _path_to_key((r.stdout or "").strip()) if getattr(r, "returncode", 1) == 0 else None
    _curkey_cache["ts"], _curkey_cache["key"] = now, key
    return key


def _sync_state():
    """Lightweight poll target: the latest run result (from ANY front-end, incl. an
    nvim `pp`) plus the editor's current file, so the browser stays in lockstep."""
    return {"run": _read_last_run(), "current": _current_nvim_key()}


def _run_item(key):
    r = _resolve(key)
    if not r:
        return {"status": "error", "message": "unknown item"}
    if not _run_lock.acquire(blocking=False):
        return {"status": "error", "message": "A run is already in progress \u2014 wait for it to finish."}
    try:
        if not nvim_save(r["path"]):
            return {"status": "error",
                    "message": "Could not save the editor buffers \u2014 is the nvim terminal connected? Reconnect the tab and retry."}
        cmd = [PYTHON, os.path.join(ROOT, "check.py"), r["pid"]] if r["kind"] == "problem" else [PYTHON, r["path"]]
        env = dict(os.environ, PREP_JSON="1")
        try:
            # new session so a timeout kills the whole tree (check.py spawns grandchildren)
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, start_new_session=True)
        except Exception as e:
            return {"status": "error", "message": f"Failed to launch run: {e}"}
        try:
            out, err = proc.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
            return {"status": "error", "message": "Run timed out after 600 s and was killed."}
        result = _parse_json_line(subprocess.CompletedProcess(cmd, proc.returncode, out, err))
        try:
            with open(LAST_RUN) as f:
                result["_ts"] = json.load(f).get("ts")
        except Exception:
            pass
        return result
    finally:
        _run_lock.release()


def _text(cmd):
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    return (proc.stdout or "") + (proc.stderr or "")


def _hint_text(key):
    r = _resolve(key)
    if not r:
        return "unknown item"
    if r["kind"] == "problem":
        return _text([PYTHON, os.path.join(ROOT, "check.py"), r["pid"], "--hint"])
    return ("Project steps don't have graded hints — read the part description under "
            "the title, or use Solution.")


def _solution_text(key, give_up):
    r = _resolve(key)
    if not r:
        return "unknown item"
    extra = ["--solution"] + (["--i-give-up"] if give_up else [])
    if r["kind"] == "problem":
        return _text([PYTHON, os.path.join(ROOT, "check.py"), r["pid"], *extra])
    return _text([PYTHON, os.path.join(ROOT, "projects.py"), r["name"], r["sid"], *extra])


def _reset_to_stub(key):
    """Restore a step/problem working file to its committed pristine stub (discards
    the user's solution for that item) and reload it in the editor."""
    r = _resolve(key)
    if not r:
        return {"ok": False, "error": "unknown item"}
    rel = os.path.relpath(r["path"], ROOT)
    try:
        proc = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10)
        if proc.returncode != 0:
            return {"ok": False, "error": "no committed stub for this item"}
        with open(r["path"], "w") as f:
            f.write(proc.stdout)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    nvim_open(r["path"])
    _nvim("--remote-send", "<C-\\><C-N>:edit!<CR>", timeout=4)
    return {"ok": True}


def _reset_progress(data):
    """Clear solved/attempted state at item / project / problems / global scope.
    Code files are untouched (that's /api/reset); notes are kept."""
    from lib import store
    scope = str(data.get("scope", ""))
    if scope == "item":
        r = _resolve(str(data.get("key", "")))
        if not r:
            return {"ok": False, "error": "unknown item"}
        if r["kind"] == "problem":
            store.relock_problem(r["pid"])
        else:
            store.relock_project_step(r["dir"], r["sid"])
    elif scope == "project":
        name = str(data.get("project", ""))
        d = os.path.join(ROOT, "projects", name)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or \
                not os.path.isfile(os.path.join(d, "project.json")):
            return {"ok": False, "error": "unknown project"}
        store.reset_project(d)
    elif scope == "problems":
        store.reset_all_problems()
    elif scope == "all":
        store.reset_everything(_project_dirs())
    else:
        return {"ok": False, "error": "unknown scope"}
    _catalog_cache["sig"] = None   # solved flags are baked into the cached catalog
    return {"ok": True}


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel):
        path = os.path.normpath(os.path.join(STATIC, rel.lstrip("/")))
        if not path.startswith(STATIC) or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = {".html": "text/html", ".js": "application/javascript",
                 ".css": "text/css"}.get(os.path.splitext(path)[1], "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _body_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            n = 0
        n = max(0, min(n, 1 << 20))   # floor at 0 (no negative read -> no hang), cap at 1 MiB
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def _authed(self):
        """Block CSRF: a malicious page in another tab can POST to loopback, but
        cannot read our same-origin /api/config token, and its Origin won't match."""
        if self.headers.get("X-MLE-Token") != _SECRET:
            return False
        origin = self.headers.get("Origin") or ""
        if not origin:
            return True   # non-browser / same-origin fetch without Origin
        host = urlparse(origin).hostname
        return host in ("127.0.0.1", "localhost", "::1")   # any loopback port (tunnel-friendly); token is the real guard

    def do_GET(self):
        try:
            return self._do_GET()
        except Exception as e:
            try: self._send(500, {"status": "error", "error": str(e)[:2000]})
            except Exception: pass

    def _do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if u.path.startswith("/static/"):
            return self._serve_static(u.path[len("/static/"):])
        if u.path == "/api/config":
            return self._send(200, {"ttyd_port": TTYD_PORT, "token": _SECRET})
        if u.path == "/api/catalog":
            return self._send(200, _catalog())
        if u.path == "/api/item":
            meta = _item_meta((q.get("key") or [""])[0])
            return self._send(200 if meta else 404, meta or {"error": "no such item"})
        if u.path == "/api/hint":
            return self._send(200, {"text": _hint_text((q.get("key") or [""])[0])})
        if u.path == "/api/sync":
            return self._send(200, _sync_state())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            return self._do_POST()
        except Exception as e:
            try: self._send(500, {"status": "error", "error": str(e)[:2000]})
            except Exception: pass

    def _do_POST(self):
        u = urlparse(self.path)
        if not self._authed():
            return self._send(403, {"error": "forbidden (missing/invalid token)"})
        data = self._body_json()
        key = str(data.get("key", ""))
        if u.path == "/api/open":
            r = _resolve(key)
            if not r:
                return self._send(404, {"error": "no such item"})
            return self._send(200, {"ok": nvim_open(r["path"])})
        if u.path == "/api/reset":
            return self._send(200, _reset_to_stub(key))
        if u.path == "/api/reset_progress":
            return self._send(200, _reset_progress(data))
        if u.path == "/api/run":
            return self._send(200, _run_item(key))
        if u.path == "/api/solution":
            return self._send(200, {"text": _solution_text(key, bool(data.get("give_up")))})
        if u.path == "/api/shutdown":
            self._send(200, {"ok": True})
            teardown()
            return
        return self._send(404, {"error": "not found"})


def _warm_up():
    """Pre-load the slow imports (torch ~1.3s) and the catalog OFF the request path,
    so the first problem you open renders instantly instead of paying import cost."""
    try:
        _catalog()                       # build + cache the picker
        import torch  # noqa: F401       # the big one — keep it off the first /api/item
        from lib import frameworks, examples, concepts  # noqa: F401
    except Exception:
        pass


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", APIPORT), Handler)
    threading.Thread(target=_warm_up, daemon=True).start()
    print(f"API on http://127.0.0.1:{APIPORT}  (nvim socket {NVIM_SOCK}, ttyd :{TTYD_PORT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
