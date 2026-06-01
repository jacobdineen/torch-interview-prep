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
inner="cd '$REPO' && export PATH='$REPO/.venv/bin':\"\$PATH\""
nvim_cmd="exec nvim"
if [ -n "${NVIM_LISTEN:-}" ]; then
  # ttyd spawns a fresh nvim per browser connection; clear any stale socket from
  # a previous (now-dead) connection first so --listen always binds cleanly.
  inner="$inner && rm -f '$NVIM_LISTEN'"
  nvim_cmd="$nvim_cmd --listen '$NVIM_LISTEN'"
fi
if [ -n "$START_FILE" ]; then
  nvim_cmd="$nvim_cmd '$START_FILE'"
fi
inner="$inner && $nvim_cmd"

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
  -t fontSize=15 -t 'titleFixed=mle_prep — nvim' "${creds[@]}" \
  bash -lc "$inner"
