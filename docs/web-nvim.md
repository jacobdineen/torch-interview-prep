# Browser Neovim (web app, phase 1)

Run **your real Neovim** — this repo, your `~/.config/nvim/init.lua`, and the
project `.venv` — in a browser tab. It is the same editor as local: every
`<leader>p` map, the winbar cheatsheet, Telescope, and the `PREP_JSON` PASS/FAIL
floats all work, because the browser is just a terminal (xterm.js) attached to a
real `nvim` process on the host via [ttyd](https://github.com/tsl0922/ttyd).

This is the terminal-first, self-host, single-user MVP. There is no second
editor and no vim *emulation* to drift from your config — it's your nvim.

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
