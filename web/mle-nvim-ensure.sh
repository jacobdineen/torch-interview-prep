#!/usr/bin/env bash
# Ensure exactly ONE persistent headless nvim server is listening on $1, starting
# it (and reaping any stale prior server) if needed. Idempotent and safe to call
# from every ttyd (re)connection. Args: <socket> [start-file]
#
# Writing the PID from inside `sh -c 'echo $$'` captures nvim's real PID (after
# exec), which `setsid ... & echo $!` does not reliably do when the caller is a
# process-group leader.
set -u
SOCK="${1:?socket required}"; START_FILE="${2:-}"
PIDFILE="/tmp/mle_nvim_server.pid"
LOG="/tmp/mle_nvim_server.log"

command -v nvim >/dev/null 2>&1 || { echo "nvim not found" >&2; exit 1; }
# already up?
nvim --server "$SOCK" --remote-expr 1 >/dev/null 2>&1 && exit 0
# reap a dead/stale prior server so we never leak nvims across :qa + reconnect
[ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null || true
rm -f "$SOCK"
if [ -n "$START_FILE" ] && [ -f "$START_FILE" ]; then
  setsid sh -c 'echo $$ > "$1"; exec nvim --headless --listen "$2" "$3"' _ "$PIDFILE" "$SOCK" "$START_FILE" >"$LOG" 2>&1 &
else
  setsid sh -c 'echo $$ > "$1"; exec nvim --headless --listen "$2"' _ "$PIDFILE" "$SOCK" >"$LOG" 2>&1 &
fi
for _ in $(seq 1 100); do nvim --server "$SOCK" --remote-expr 1 >/dev/null 2>&1 && exit 0; sleep 0.1; done
echo "nvim server did not come up within 10s" >&2
exit 1
