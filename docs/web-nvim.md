# Web app — practice in the browser, edited in your real Neovim

Two ways to run this in a browser, both self-host / single-user, both using your
**real Neovim** (this repo, your `~/.config/nvim/init.lua`, the project `.venv`).
It's the same editor as local — every `<leader>p` map, the winbar, Telescope, and
the `PREP_JSON` floats work, because the browser just renders a real `nvim`
process via [ttyd](https://github.com/tsl0922/ttyd). No vim *emulation* to drift.

1. **Full app** (`web/serve-app.sh`) — a problem-description UI on the left, your
   nvim embedded on the right, with Run/Submit and a results panel. Recommended.
2. **Just the editor** (`web/serve-nvim.sh`) — only the nvim terminal in a tab.

## Full app

```bash
sudo apt-get install -y ttyd && sudo systemctl disable --now ttyd   # one-time (see note below)
./web/serve-app.sh
```

Open **http://127.0.0.1:8000**. Pick a problem on the left (or use *next ›*); it
opens in the embedded nvim via `nvim --remote` (one persistent nvim — switching
problems just changes the buffer). Edit as you always do, then hit **Run**: the
app remote-saves all buffers (`:wa`) and runs `check.py`, rendering PASS/FAIL +
the likely cause + the tensor diff in the results panel. **Submit** also shows
the concept blurb on success. **Hint**/**Solution** call the same CLI helpers.

How it fits together:

```
browser ──HTTP──► web/app.py (:8000)  ── nvim --remote / --remote-send ──► nvim
   │                  └─ runs check.py (PREP_JSON) for Run/Submit            ▲
   └──iframe──► ttyd (:7681) ──PTY──► nvim --listen /tmp/mle_nvim.sock ──────┘
```

**Stopping it.** Click **⏻ Tear down** in the top bar (saves + quits nvim, stops
ttyd, stops the server), or `Ctrl-C` the launcher, or run `./web/teardown.sh`.

Both ports bind **loopback only**. Env knobs: `APIPORT` (default 8000),
`TTYD_PORT` (7681), `NVIM_SOCK` (`/tmp/mle_nvim.sock`). To reach a remote box,
tunnel **both** ports:

```bash
ssh -L 8000:localhost:8000 -L 7681:localhost:7681 <you>@<this-host>
```

## Just the editor

```bash
./web/serve-nvim.sh
```

Run **your real Neovim** in a browser tab (no surrounding UI). Same editor,
binds loopback, opens to your next unsolved problem.

## One-time setup

```bash
sudo apt-get install -y ttyd        # Ubuntu/Debian (1.7.x in universe)
```

> The apt package installs and **enables** a `ttyd.service` that exposes a login
> shell on `0.0.0.0:7681`. Disable it so nothing is served unexpectedly — you
> start ttyd yourself with the launcher below:
>
> ```bash
> sudo systemctl disable --now ttyd
> ```

## Run it

```bash
./web/serve-nvim.sh
```

Binds to the **loopback interface only** and opens straight to your next unsolved
problem. Then open `http://127.0.0.1:7681` in a browser on the same machine, or
reach it from elsewhere with an SSH tunnel:

```bash
ssh -L 7681:localhost:7681 <you>@<this-host>
# now browse to http://localhost:7681 on your laptop
```

Stop the server with `Ctrl-C` in the terminal where it runs.

### Knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `PORT` | `7681` | Port to listen on. |
| `BIND` | `lo` | Network interface to bind. `lo` = loopback only. Set to e.g. `tailscale0` to expose on your tailnet. |
| `AUTH` | _(none)_ | HTTP basic auth `user:password`. **Required** whenever `BIND` is not `lo`. |

Examples:

```bash
PORT=8080 ./web/serve-nvim.sh
BIND=tailscale0 AUTH=me:s3cret ./web/serve-nvim.sh   # reachable across your tailnet, behind auth
```

## Security notes

- **Default is loopback + no exposure.** Reach it via SSH tunnel or Tailscale.
- A browser terminal is a **shell on this machine**. Never bind to a public
  interface (`0.0.0.0`, your LAN IP) without auth, and prefer TLS (ttyd `--ssl`,
  or a reverse proxy like Caddy) and a network you trust (Tailscale).
- The launcher runs a single client (`-m 1`) with origin checking (`-O`).

## Why this design (and what's next)

"Exactly the same as local nvim" rules out an in-browser editor with vim
emulation (Monaco/CodeMirror): your `init.lua`, Lua plugins, and the entire
`<leader>p` integration wouldn't run there. Streaming a real `nvim` over a
terminal is the only faithful option.

Possible later phases (not built yet): a web shell *around* the terminal (the
`prep.py` dashboard and problem picker as HTML), multi-user with per-user
sandboxed containers, and a hosted deployment. Those only make sense beyond a
single self-hosted user — see the audience trade-offs before taking them on.
