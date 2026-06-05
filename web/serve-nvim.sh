#!/usr/bin/env bash
# Serve your REAL Neovim — this repo, your ~/.config/nvim, and the project's
# .venv — in a browser via ttyd (xterm.js over a WebSocket PTY). It's the same
# nvim you run locally: every <leader>p map, the winbar, Telescope, and the
# PREP_JSON floats work because it IS your nvim, just rendered in the browser.
#
# Terminal-first, self-host, single user. See docs/web-nvim.md.
#
#   ./web/serve-nvim.sh                 # bind loopback only (reach via SSH tunnel / Tailscale)
#   PORT=8080 ./web/serve-nvim.sh       # different port
#   BIND=tailscale0 AUTH=me:secret ./web/serve-nvim.sh   # expose on Tailscale, with basic auth
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${PORT:-7681}"
BIND="${BIND:-lo}"     # loopback interface by default; set to an iface name (e.g. tailscale0) to expose
AUTH="${AUTH:-}"       # optional "user:password" for HTTP basic auth (use when BIND is not lo)

if ! command -v ttyd >/dev/null 2>&1; then
  echo "ttyd not found. Install it (Ubuntu/Debian): sudo apt-get install -y ttyd" >&2
  exit 1
fi
if ! command -v nvim >/dev/null 2>&1; then
  echo "nvim not found on PATH." >&2
  exit 1
fi

# Open straight to the next unsolved problem when we can; otherwise plain nvim.
START_FILE=""
if [ -x "$REPO/.venv/bin/python" ]; then
  START_FILE="$("$REPO/.venv/bin/python" "$REPO/check.py" --next 2>/dev/null | tail -1 || true)"
fi
[ -f "${START_FILE:-}" ] || START_FILE=""

# Build the command the PTY runs: enter the repo, put the venv first on PATH (so
# `python`/`uv` resolve to the project's), then exec your real nvim.
# NVIM_LISTEN (optional): start nvim with a --listen socket so the web app can
# drive it remotely (open buffers, save) — see web/serve-app.sh.
SOCK="${NVIM_LISTEN:-}"
PERSIST="${NVIM_PERSISTENT:-1}"          # 1 = ONE persistent server + a UI client per connection (default); 0 = legacy fresh-nvim-per-connection
SRV_PIDFILE="/tmp/mle_nvim_server.pid"
SRV_LOG="/tmp/mle_nvim_server.log"

# Ensure exactly ONE persistent headless nvim is listening on $SOCK, (re)starting
# it if needed. Idempotent; safe under ttyd -m 1. This is the fix for the old race
# where every browser (re)connection spawned a new nvim that rm-ed and rebound the
# shared socket, orphaning the others and leaving a stale, unresponsive socket.
ensure_server() {
  nvim --server "$SOCK" --remote-expr 1 >/dev/null 2>&1 && return 0
  rm -f "$SOCK"
  if [ -n "$START_FILE" ]; then
    setsid nvim --headless --listen "$SOCK" "$START_FILE" >"$SRV_LOG" 2>&1 &
  else
    setsid nvim --headless --listen "$SOCK" >"$SRV_LOG" 2>&1 &
  fi
  echo $! > "$SRV_PIDFILE"
  for _ in $(seq 1 100); do nvim --server "$SOCK" --remote-expr 1 >/dev/null 2>&1 && return 0; sleep 0.1; done
  return 1
}

if [ -n "$SOCK" ] && [ "$PERSIST" = "1" ]; then
  ensure_server || echo "warning: nvim server slow to start; reconnect if the editor looks blank" >&2
  ESC_START=""; [ -n "$START_FILE" ] && ESC_START=" '$START_FILE'"
  # Per ttyd connection: attach a UI to the persistent server, reviving it first if
  # it ever exited (e.g. the user ran :qa). Buffers survive reconnects; no race.
  inner="cd '$REPO' && export PATH='$REPO/.venv/bin':\"\$PATH\"
if ! nvim --server '$SOCK' --remote-expr 1 >/dev/null 2>&1; then rm -f '$SOCK'; setsid nvim --headless --listen '$SOCK'$ESC_START >'$SRV_LOG' 2>&1 & echo \$! > '$SRV_PIDFILE'; for _ in \$(seq 1 100); do nvim --server '$SOCK' --remote-expr 1 >/dev/null 2>&1 && break; sleep 0.1; done; fi
exec nvim --server '$SOCK' --remote-ui"
else
  # legacy: a fresh nvim per connection (set NVIM_PERSISTENT=0 to force this)
  inner="cd '$REPO' && export PATH='$REPO/.venv/bin':\"\$PATH\""
  nvim_cmd="exec nvim"
  if [ -n "$SOCK" ]; then inner="$inner && rm -f '$SOCK'"; nvim_cmd="$nvim_cmd --listen '$SOCK'"; fi
  [ -n "$START_FILE" ] && nvim_cmd="$nvim_cmd '$START_FILE'"
  inner="$inner && $nvim_cmd"
fi

creds=()
[ -n "$AUTH" ] && creds=(-c "$AUTH")

if [ "$BIND" = "lo" ]; then
  echo "Serving on http://127.0.0.1:$PORT (loopback only)."
  echo "Reach it from another machine via an SSH tunnel:"
  echo "    ssh -L $PORT:localhost:$PORT $(whoami)@<this-host>"
  echo "or set BIND=tailscale0 (and AUTH=user:pass) to expose on Tailscale. See docs/web-nvim.md."
elif [ -z "$AUTH" ]; then
  echo "WARNING: binding to interface '$BIND' with NO auth — anyone who can reach it gets your shell." >&2
  echo "         Set AUTH=user:password (and ideally TLS). Continuing in 3s; Ctrl-C to abort." >&2
  sleep 3
fi

# -W writable, -O check-origin, -m 1 single client.
exec ttyd -W -O -m 1 -i "$BIND" -p "$PORT" \
  -t fontSize="${FONT_SIZE:-15}" -t 'titleFixed=mle_prep — nvim' "${creds[@]}" \
  bash -lc "$inner"
