"use strict";

const $ = (id) => document.getElementById(id);
const api = {
  async get(p) { const r = await fetch(p); if (!r.ok) throw new Error(`GET ${p} → ${r.status}`); return r.json(); },
  async post(p, body) {
    const send = () => fetch(p, { method: "POST", headers: { "Content-Type": "application/json", "X-MLE-Token": TOKEN }, body: JSON.stringify(body || {}) });
    let r = await send();
    if (r.status === 403) {                 // stale token (server restarted) -> refresh it and retry once
      try { const cfg = await (await fetch("/api/config")).json(); TOKEN = cfg.token || ""; }
      catch (e) { console.warn("token refresh failed (server down?)", e); }
      r = await send();
    }
    if (!r.ok) throw new Error(`POST ${p} → ${r.status}`); return r.json();
  },
};

let ITEMS = [], SOURCES = [], PROJECTS = {}, TOKEN = "";
let CURRENT = null, HILITE = 0;
let _selSeq = 0, _runSeq = 0, _runStatusTimer = null;
let _lastRunTs = 0;      // ts of the most recently handled run (web or nvim) — dedups the /api/sync poll
let _runInFlight = 0;    // # of web runs in flight — poll must not also react to them
let _syncSeeded = false; // first poll only establishes a baseline (no replay)

async function init() {
  applyTheme(localStorage.getItem("mle_theme") || "dark");
  applyZoom(parseFloat(localStorage.getItem("mle_ed_zoom")) || 1);

  const cfg = await api.get("/api/config");
  TOKEN = cfg.token || "";
  $("nvim").src = `${location.protocol}//${location.hostname}:${cfg.ttyd_port}/`;

  const cat = await api.get("/api/catalog");
  ITEMS = cat.items || []; SOURCES = cat.sources || []; PROJECTS = cat.projects || {};

  const sourceSel = $("source");
  sourceSel.innerHTML = "";
  for (const s of ["All", ...SOURCES]) {
    const o = document.createElement("option");
    o.value = s; o.textContent = s === "All" ? "All sources" : s; sourceSel.appendChild(o);
  }
  sourceSel.value = "Problems";

  const fwSel = $("framework");
  fwSel.innerHTML = "";
  const fwLabel = { numpy: "NumPy", torch: "PyTorch" };
  for (const fw of ["All", ...(cat.frameworks || ["numpy", "torch"])]) {
    const o = document.createElement("option");
    o.value = fw; o.textContent = fw === "All" ? "Any framework" : (fwLabel[fw] || fw); fwSel.appendChild(o);
  }
  fwSel.value = "All";
  fwSel.addEventListener("change", refreshView);

  setupSplitters();
  setupShortcuts();
  setupPalette();
  window.addEventListener("resize", debounce(clampSplits, 120));
  window.addEventListener("popstate", onPopState);

  sourceSel.addEventListener("change", refreshView);
  const f = $("filter");
  f.addEventListener("input", refreshView);
  f.addEventListener("focus", () => { renderPalette(); openPalette(); });
  f.addEventListener("keydown", onFilterKey);
  document.addEventListener("click", (e) => { if (!e.target.closest(".search-wrap")) closePalette(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("outline").classList.contains("hidden")) closeOutline(); });

  $("home-btn").addEventListener("click", showHome);
  $("theme-btn").addEventListener("click", toggleTheme);
  $("zoom-in").addEventListener("click", () => bumpZoom(+0.1));
  $("zoom-out").addEventListener("click", () => bumpZoom(-0.1));
  $("prev-btn").addEventListener("click", () => step(-1));
  $("next-btn").addEventListener("click", gotoNextUnsolved);
  $("run-btn").addEventListener("click", () => run(false));
  $("submit-btn").addEventListener("click", () => run(true));
  $("hint-btn").addEventListener("click", showHint);
  $("solution-btn").addEventListener("click", showSolution);
  $("teardown-btn").addEventListener("click", teardown);
  $("status-filter").addEventListener("change", refreshView);
  $("outline-btn").addEventListener("click", toggleOutline);
  $("sidebar-toggle").addEventListener("click", toggleSidebar);
  $("sidebar-collapse").addEventListener("click", toggleSidebar);
  if (localStorage.getItem("mle_sidebar_collapsed") === "1") document.body.classList.add("sidebar-collapsed");
  $("outline-close").addEventListener("click", closeOutline);
  $("outline-scrim").addEventListener("click", closeOutline);
  $("reset-btn").addEventListener("click", resetToStub);
  $("reset-progress-btn").addEventListener("click", openResetDlg);
  $("rd-close").addEventListener("click", () => $("reset-dlg").close());
  $("reset-dlg").addEventListener("click", (e) => { if (e.target.id === "reset-dlg") $("reset-dlg").close(); });
  $("autoadvance").checked = localStorage.getItem("mle_autoadvance") === "1";
  $("autoadvance").addEventListener("change", (e) => localStorage.setItem("mle_autoadvance", e.target.checked ? "1" : "0"));
  $("cmdk-input").addEventListener("input", () => { CMDK_HI = 0; renderCmdk(); });
  $("cmdk-input").addEventListener("keydown", onCmdkKey);
  $("cmdk").addEventListener("click", (e) => { if (e.target.id === "cmdk") $("cmdk").close(); });
  $("sd-close").addEventListener("click", () => $("shortcuts-dlg").close());
  $("shortcuts-dlg").addEventListener("click", (e) => { if (e.target.id === "shortcuts-dlg") $("shortcuts-dlg").close(); });

  setInterval(syncPoll, 1200);   // syncPoll() baselines itself on its first tick

  const deep = new URLSearchParams(location.search).get("key");
  const deepItem = deep ? ITEMS.find((x) => x.key === deep) : null;
  if (deepItem) enterTrack(deepItem.source, deepItem.framework || "All", deepItem.key);
  else showHome();
}

// Any filter/source/status change: re-render the palette + progress for the new view.
function refreshView() { HILITE = 0; renderPalette(); openPalette(); updateProgress(); }

function showInitError(err) {
  console.error(err);
  const el = document.createElement("div");
  el.id = "init-error";
  el.innerHTML = `<div class="ie-card"><h2>Couldn't reach the server</h2>` +
    `<p>${esc((err && err.message) || "Network error")}. The API on this port may be down — is <code>serve-app.sh</code> still running?</p>` +
    `<button onclick="location.reload()">Retry</button></div>`;
  document.body.appendChild(el);
}
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function inField() {
  const a = document.activeElement;
  return /^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(a.tagName) || !!(a.closest && a.closest("details summary"));
}

// ===== theme =====
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  $("theme-btn").textContent = t === "light" ? "☀" : "☾";
  $("theme-btn").title = t === "light" ? "Switch to dark" : "Switch to light";
}
function toggleTheme() {
  const t = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  localStorage.setItem("mle_theme", t); applyTheme(t);
}

// ===== editor zoom =====
const ZMIN = 0.7, ZMAX = 2.2;
function applyZoom(z) {
  z = Math.max(ZMIN, Math.min(ZMAX, Math.round(z * 100) / 100));
  document.documentElement.style.setProperty("--ed-zoom", z);
  const lbl = $("zoom-val"); if (lbl) lbl.textContent = Math.round(z * 100) + "%";
  localStorage.setItem("mle_ed_zoom", z);
}
function curZoom() { return parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--ed-zoom")) || 1; }
function bumpZoom(d) { applyZoom(curZoom() + d); }

// ===== home / track picker =====
function trackStats(pred) { const v = ITEMS.filter(pred); return { total: v.length, solved: v.filter((x) => x.solved).length }; }
function card(cls, icon, title, sub, desc, stats, onClick) {
  const pct = stats.total ? Math.round((stats.solved / stats.total) * 100) : 0;
  const complete = stats.total > 0 && pct === 100;
  const el = document.createElement("button");
  el.className = "home-card " + cls + (complete ? " complete" : "");
  el.innerHTML =
    `<div class="hc-top"><div class="hc-icon">${icon}</div>` +
    `<div><div class="hc-title">${esc(title)}</div><div class="hc-sub">${esc(sub)}</div></div></div>` +
    `<div class="hc-desc">${esc(desc)}</div>` +
    `<div class="hc-bar"><div style="width:${pct}%"></div></div>` +
    `<div class="hc-meta"><span>${stats.solved}/${stats.total} solved</span>` +
    `<span>${complete ? "✓ complete" : pct + "%"}</span></div>`;
  el.addEventListener("click", onClick);
  return el;
}
function renderHome() {
  renderStats();
  const rb = $("resume-banner"); rb.innerHTML = "";
  const lastKey = localStorage.getItem("mle_last_key");
  const last = lastKey && ITEMS.find((x) => x.key === lastKey);
  if (last) {
    rb.classList.remove("hidden");
    rb.innerHTML = `<div><div class="rb-label">Resume</div>` +
      `<div class="rb-title">${esc(last.title)}</div>` +
      `<div class="rb-sub">${esc(last.source === "Problems" ? "" : last.source + " · ")}${esc(last.id)}${last.solved ? " · solved" : ""}</div></div>`;
    const btn = document.createElement("button"); btn.textContent = "Continue →";
    btn.addEventListener("click", () => enterTrack(last.source, last.framework || "All", last.key));
    rb.appendChild(btn);
  } else rb.classList.add("hidden");

  const probs = $("home-problems"); probs.innerHTML = "";
  if (!ITEMS.length) { probs.innerHTML = `<div class="hc-desc">No items loaded — the catalog came back empty.</div>`; return; }
  const isProb = (x) => x.source === "Problems";
  probs.appendChild(card("torch", "🔥", "PyTorch Problems", "Tensors, autograd, nn, losses",
    "Core deep-learning building blocks implemented in PyTorch.",
    trackStats((x) => isProb(x) && x.framework === "torch"), () => enterTrack("Problems", "torch")));
  probs.appendChild(card("numpy", "▦", "NumPy Problems", "Same problems, pure NumPy",
    "Everything that has a NumPy variant — no autograd, just arrays.",
    trackStats((x) => isProb(x) && (x.framework === "numpy" || x.numpy)), () => enterTrack("Problems", "numpy")));
  probs.appendChild(card("all", "∑", "All Problems", "Every standalone problem",
    "Browse the whole problem set across both frameworks.",
    trackStats(isProb), () => enterTrack("Problems", "All")));

  const projs = $("home-projects"); projs.innerHTML = "";
  for (const name of SOURCES.filter((s) => s !== "Problems")) {
    const items = ITEMS.filter((x) => x.source === name);
    const parts = new Set(items.map((x) => x.group)).size;
    const meta = PROJECTS[name] || {};
    projs.appendChild(card("project", "📦", meta.title || name, `${parts} part${parts === 1 ? "" : "s"} · ${items.length} steps`,
      meta.description || "Build it end-to-end, one graded step at a time.",
      trackStats((x) => x.source === name), () => enterTrack(name, "All")));
  }
}
function showHome() {
  _selSeq++;                                    // cancel any in-flight selectItem
  document.body.classList.add("home-active");
  $("split").classList.add("hidden");
  $("home").classList.remove("hidden");
  closePalette(); renderHome();
  try { history.replaceState({}, "", location.pathname); } catch (e) {}
  const h = $("home").querySelector("h1"); if (h) h.focus();
}
function enterTrack(source, fw, key) {
  document.body.classList.remove("home-active");
  $("home").classList.add("hidden");
  $("split").classList.remove("hidden");
  if ([...$("source").options].some((o) => o.value === source)) $("source").value = source;
  if ([...$("framework").options].some((o) => o.value === fw)) $("framework").value = fw;
  localStorage.setItem("mle_last_track", JSON.stringify({ source, fw }));
  renderPalette(); updateProgress(); clampSplits();
  const v = view();
  const target = key ? (v.find((x) => x.key === key) || ITEMS.find((x) => x.key === key))
    : (v.find((x) => !x.solved) || v[0]);
  if (target) selectItem(target.key);
}

// ===== progress =====
function updateProgress() {
  const v = view(); const solved = v.filter((x) => x.solved).length;
  const pct = v.length ? Math.round((solved / v.length) * 100) : 0;
  $("progress-fill").style.width = pct + "%";
  $("overall").textContent = `${solved}/${v.length} solved`;
}

// ===== resizable panes =====
function setupSplitters() {
  const root = document.documentElement;
  const restore = (k, v) => { const s = localStorage.getItem(k); if (s) root.style.setProperty(v, s); };
  restore("mle_left_w", "--left-w"); restore("mle_results_h", "--results-h");

  const drag = (gutter, axisClass, compute, storeKey, cssVar) => {
    if (!gutter) return;
    gutter.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      try { gutter.setPointerCapture(e.pointerId); } catch (_) {}
      gutter.classList.add("dragging");
      document.body.classList.add("resizing", axisClass);
      const move = (ev) => root.style.setProperty(cssVar, compute(ev) + "px");
      const end = (ev) => {
        gutter.classList.remove("dragging");
        document.body.classList.remove("resizing", axisClass);
        gutter.removeEventListener("pointermove", move);
        gutter.removeEventListener("pointerup", end);
        gutter.removeEventListener("pointercancel", end);
        try { gutter.releasePointerCapture((ev && ev.pointerId != null) ? ev.pointerId : e.pointerId); } catch (_) {}
        localStorage.setItem(storeKey, getComputedStyle(root).getPropertyValue(cssVar).trim());
      };
      gutter.addEventListener("pointermove", move);
      gutter.addEventListener("pointerup", end);
      gutter.addEventListener("pointercancel", end);
    });
  };
  const split = $("split"), right = $("right");
  drag($("gutter-x"), "cols", (ev) => {
    const r = split.getBoundingClientRect(), sw = sidebarWidth();
    return Math.max(280, Math.min(r.width - sw - 360, ev.clientX - r.left - sw));
  }, "mle_left_w", "--left-w");
  drag($("gutter-y"), "rows", (ev) => {
    const r = right.getBoundingClientRect();
    return Math.max(70, Math.min(r.height - 150, r.bottom - ev.clientY));
  }, "mle_results_h", "--results-h");
  clampSplits();
}
function sidebarWidth() {
  const sb = document.getElementById("sidebar");
  return (sb && getComputedStyle(sb).display !== "none") ? sb.getBoundingClientRect().width : 0;
}

function toggleSidebar() {
  document.body.classList.toggle("sidebar-collapsed");
  localStorage.setItem("mle_sidebar_collapsed",
    document.body.classList.contains("sidebar-collapsed") ? "1" : "0");
  clampSplits();
}

function clampSplits() {
  if (document.body.classList.contains("home-active")) return;
  const root = document.documentElement, split = $("split"), left = $("left"), right = $("right");
  if (!split || !left) return;
  const sw = split.getBoundingClientRect().width, sbw = sidebarWidth();
  if (sw > 0) {
    const lw = Math.max(280, Math.min(sw - sbw - 360, left.getBoundingClientRect().width));
    root.style.setProperty("--left-w", Math.round(lw) + "px");
  }
  const rh = right.getBoundingClientRect().height;
  if (rh > 0) {
    const cur = parseFloat(getComputedStyle(root).getPropertyValue("--results-h")) || 230;
    root.style.setProperty("--results-h", Math.round(Math.max(70, Math.min(rh - 180, cur))) + "px");
  }
}

function setupShortcuts() {
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) { e.preventDefault(); openCmdk(); return; }
    const field = inField();
    const home = document.body.classList.contains("home-active");
    // editor zoom: only in the workspace, page-focused
    if ((e.ctrlKey || e.metaKey) && !field && !home) {
      if (e.key === "=" || e.key === "+") { e.preventDefault(); bumpZoom(+0.1); return; }
      if (e.key === "-" || e.key === "_") { e.preventDefault(); bumpZoom(-0.1); return; }
      if (e.key === "0") { e.preventDefault(); applyZoom(1); return; }
    }
    if (field || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "?") { e.preventDefault(); $("shortcuts-dlg").showModal(); return; }
    if (e.key === "/" && !home) { e.preventDefault(); $("filter").focus(); $("filter").select(); return; }
    if (home) return;
    if (e.key === "r") { e.preventDefault(); run(false); }
    else if (e.key === "s") { e.preventDefault(); run(true); }
    else if (e.key === "n") { e.preventDefault(); gotoNextUnsolved(); }
    else if (e.key === "h") { e.preventDefault(); showHint(); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
    else if (e.key === "g") { e.preventDefault(); showHome(); }
    else if (e.key === "o") { e.preventDefault(); toggleOutline(); }
  });
}

// ===== filtered view =====
function view() {
  const src = $("source").value, fw = $("framework").value, q = $("filter").value.trim().toLowerCase();
  return ITEMS.filter((x) => {
    if (src !== "All" && x.source !== src) return false;
    if (fw !== "All" && x.framework !== fw && !(fw === "numpy" && x.numpy)) return false;
    if (!statusOk(x)) return false;
    if (!q) return true;
    return (x.id + " " + x.title + " " + x.group).toLowerCase().includes(q);
  });
}
function fwShort(it) { if (it.framework === "numpy") return "np"; return it.numpy ? "pt+np" : "pt"; }

// ===== palette =====
const PALETTE_MAX = 250;          // cap rows actually rendered; refine search for more
let RENDERED = [];                // the items currently in the palette (== view, capped)
function renderPalette() {
  const v = view(), pal = $("palette");
  RENDERED = v.length > PALETTE_MAX ? v.slice(0, PALETTE_MAX) : v;
  if (HILITE >= RENDERED.length) HILITE = Math.max(0, RENDERED.length - 1);
  if (!v.length) { pal.innerHTML = `<div class="pal-empty">no matches</div>`; return; }
  let html = "", group = null;
  for (let i = 0; i < RENDERED.length; i++) {
    const it = RENDERED[i];
    if (it.group !== group) { group = it.group; html += `<button type="button" class="pal-group" data-group="${esc(group)}" title="Jump to first unsolved here">${esc(group)}</button>`; }
    html += `<div class="pal-row${i === HILITE ? " hi" : ""}" role="option" aria-selected="${i === HILITE}" data-key="${esc(it.key)}" data-i="${i}">` +
      `<span class="mark ${it.solved ? "ok" : ""}">${it.solved ? "\u2713" : "\u00b7"}</span>` +
      `<span class="fwdot ${esc(it.framework || "")}" title="${it.numpy ? "torch + numpy" : esc(it.framework || "")}">${fwShort(it)}</span>` +
      `<span class="pid">${esc(it.id)}</span><span class="ptitle">${esc(it.title)}</span></div>`;
  }
  if (v.length > RENDERED.length) html += `<div class="pal-more">+${v.length - RENDERED.length} more \u2014 keep typing to narrow</div>`;
  pal.innerHTML = html;
  scrollHiliteIntoView();
}
// Delegate clicks/hover ONCE on the container instead of per-row on every render.
function setupPalette() {
  const pal = $("palette");
  pal.addEventListener("click", (e) => {
    const row = e.target.closest(".pal-row");
    if (row) { selectItem(row.dataset.key); closePalette(); $("filter").blur(); return; }
    const g = e.target.closest(".pal-group");
    if (g) {
      const grp = g.dataset.group, inGrp = view().filter((x) => x.group === grp);
      const t = inGrp.find((x) => !x.solved) || inGrp[0];
      if (t) { selectItem(t.key); closePalette(); $("filter").blur(); }
    }
  });
  pal.addEventListener("mousemove", (e) => {
    const row = e.target.closest(".pal-row");
    if (row) { const i = parseInt(row.dataset.i, 10); if (i !== HILITE) setHilite(i); }
  });
}
// O(1): only repaint the previously- and newly-highlighted rows.
function setHilite(i) {
  const pal = $("palette");
  const prev = pal.querySelector(".pal-row.hi");
  if (prev) { prev.classList.remove("hi"); prev.setAttribute("aria-selected", "false"); }
  HILITE = i;
  const cur = pal.querySelector(`.pal-row[data-i="${i}"]`);
  if (cur) { cur.classList.add("hi"); cur.setAttribute("aria-selected", "true"); }
}
function scrollHiliteIntoView() { const el = $("palette").querySelector(".pal-row.hi"); if (el) el.scrollIntoView({ block: "nearest" }); }
function openPalette() { $("palette").classList.remove("hidden"); $("filter").setAttribute("aria-expanded", "true"); }
function closePalette() { $("palette").classList.add("hidden"); $("filter").setAttribute("aria-expanded", "false"); }

function onFilterKey(e) {
  const v = RENDERED;
  if (e.key === "ArrowDown") { e.preventDefault(); HILITE = Math.min(v.length - 1, HILITE + 1); setHilite(HILITE); scrollHiliteIntoView(); openPalette(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); HILITE = Math.max(0, HILITE - 1); setHilite(HILITE); scrollHiliteIntoView(); }
  else if (e.key === "Enter") { e.preventDefault(); if (v[HILITE]) { selectItem(v[HILITE].key); closePalette(); $("filter").blur(); } }
  else if (e.key === "Escape") { e.preventDefault(); closePalette(); }   // keep focus in the field
}

// ===== navigation =====
function step(delta) {
  const v = view(); const i = v.findIndex((x) => x.key === CURRENT);
  const j = Math.min(v.length - 1, Math.max(0, (i < 0 ? 0 : i + delta)));
  if (v[j]) selectItem(v[j].key);
}
function gotoNextUnsolved() {
  const v = view(); const i = v.findIndex((x) => x.key === CURRENT);
  const nxt = v.slice(i + 1).find((x) => !x.solved) || v.find((x) => !x.solved);
  if (nxt) selectItem(nxt.key);
  else flashResults("muted", `All ${v.length} item${v.length === 1 ? "" : "s"} in this view are solved — switch track or filter to find more.`);
}

// ===== selection =====
function setEditorWarn(msg, kind) {
  const w = $("editor-warn"); if (!w) return;
  w.textContent = msg || ""; w.className = "editor-warn" + (msg ? " " + (kind || "warn") : "");
}
async function openEditor(key, seq) {
  setEditorWarn("opening…", "info");
  try {
    const r = await api.post("/api/open", { key });
    if (seq === _selSeq) setEditorWarn(r && r.ok ? "" : "editor may not have switched — check nvim", "warn");
  } catch (e) { if (seq === _selSeq) setEditorWarn("editor open failed", "warn"); }
}
async function selectItem(key, push = true, openNvim = true) {
  const seq = ++_selSeq;
  CURRENT = key;
  localStorage.setItem("mle_last_key", key);
  setEditorWarn("");
  $("left").classList.add("loading");
  if (openNvim) openEditor(key, seq);           // switch the editor in the background — never block the description on nvim (skip when we're FOLLOWING nvim)
  let m;
  try { m = await api.get("/api/item?key=" + encodeURIComponent(key)); }
  catch (e) { if (seq === _selSeq) { $("left").classList.remove("loading"); flashResults("fail", "Couldn't load " + key + ": " + e.message); } return; }
  if (seq !== _selSeq) return;
  $("left").classList.remove("loading");
  if (!m || m.error) return;

  $("prob-id").textContent = (m.source && m.source !== "Problems" ? m.source + " · " : "") + (m.id || "");
  $("prob-title").textContent = m.title || m.id;
  $("prob-group").textContent = m.group || "";
  setBreadcrumb(key);
  if (!$("outline").classList.contains("hidden")) renderOutline();
  const fwt = $("prob-framework");
  let fwText = m.framework === "torch" ? "PyTorch" : m.framework === "numpy" ? "NumPy" : "";
  if (m.numpy) fwText = "PyTorch + NumPy";
  fwt.textContent = fwText;
  fwt.title = m.numpy ? "Also solvable in NumPy — `check.py " + (m.id || "") + " --numpy`" : "";
  fwt.className = "tag fw " + (m.framework || "");
  const st = $("prob-status");
  st.className = "tag " + (m.solved ? "solved" : m.last_status === "fail" ? "failed" : "");
  st.textContent = m.solved ? "solved" : m.last_status === "fail" ? "attempted" : "unsolved";
  $("prob-sig").textContent = m.signature || "";
  $("prob-doc").textContent = m.doc || "";
  const ew = $("example-wrap");
  if (m.example) {
    $("ex-input").textContent = m.example.inputs || "—";
    $("ex-output").textContent = m.example.output || (m.example.matches ? "should match " + m.example.matches : "—");
    ew.style.display = "";
  } else ew.style.display = "none";
  const cw = $("concept-wrap");
  if (m.concept) { $("prob-concept").textContent = m.concept; cw.style.display = ""; } else cw.style.display = "none";
  $("aux-out").textContent = "";
  resetResults();
  if (m.source && [...$("source").options].some((o) => o.value === m.source)) $("source").value = m.source;
  try { if (push && new URLSearchParams(location.search).get("key") !== key) history.pushState({ key }, "", "?key=" + encodeURIComponent(key)); } catch (e) {}
}
function onPopState() {
  const k = new URLSearchParams(location.search).get("key");
  if (k && ITEMS.find((x) => x.key === k)) {
    if (document.body.classList.contains("home-active")) {   // came from the home view
      document.body.classList.remove("home-active");
      $("home").classList.add("hidden");
      $("split").classList.remove("hidden");
      renderPalette(); updateProgress(); clampSplits();
    }
    if (k !== CURRENT) selectItem(k, false);
  } else showHome();
}

function resetResults() {
  const b = $("results-body");
  b.className = "results-body muted";
  b.innerHTML = `<div class="results-placeholder"><div class="rp-glyph">▶</div>` +
    `<div>Write your solution in the editor</div>` +
    `<div class="rp-dim">then press <kbd>r</kbd> or click Run</div></div>`;
}
function flashResults(kind, text) { const b = $("results-body"); b.className = "results-body " + kind; b.textContent = text; }

async function run(submit) {
  const seq = ++_runSeq, selAt = _selSeq, key = CURRENT;
  if (!key) return;
  _runInFlight++;
  $("run-btn").disabled = true; $("submit-btn").disabled = true;
  const b = $("results-body"); b.className = "results-body muted";
  b.textContent = (submit ? "Submitting" : "Running") + " " + label(key) + " …";
  let r;
  try {
    try { r = await api.post("/api/run", { key }); }
    catch (e) { if (seq === _runSeq && selAt === _selSeq) flashResults("fail", "error: " + e.message); }
    if (seq !== _runSeq || selAt !== _selSeq || !r) return;   // navigated away mid-run
    renderResult(r, submit);
    setRunStatus(r.status);
    if (r._ts) _lastRunTs = r._ts;   // claim this run's ts so the poll won't replay it
    if (r.status === "pass") { markSolved(key); maybeAutoAdvance(); }
  } finally {
    if (seq === _runSeq) { $("run-btn").disabled = false; $("submit-btn").disabled = false; }
    _runInFlight--;
  }
}
function setRunStatus(status) {
  const btn = $("run-btn");
  if (status !== "pass" && status !== "fail") return;
  btn.dataset.runStatus = status;
  clearTimeout(_runStatusTimer);
  _runStatusTimer = setTimeout(() => { delete btn.dataset.runStatus; }, 3000);
}
function markSolved(key) {
  const it = ITEMS.find((x) => x.key === key);
  const newly = it && !it.solved;
  if (newly) {
    it.solved = true; updateProgress(); recordSolve(key); celebrate(key);
    if (!$("outline").classList.contains("hidden")) renderOutline();
  }
  // reflect in the palette row if present
  const row = $("palette").querySelector(`.pal-row[data-key="${cssEsc(key)}"] .mark`);
  if (row) { row.classList.add("ok"); row.textContent = "✓"; }
  if (CURRENT === key) { $("prob-status").className = "tag solved"; $("prob-status").textContent = "solved"; }
}
function renderResult(r, submit) {
  const b = $("results-body");
  if (r.status === "pass") {
    b.className = "results-body pass";
    let h = `<div class="res-headline">✓ PASS ${esc(r.problem || "")}</div>`;
    if (r.progress) h += `<div class="res-sec">${esc(r.progress)}</div>`;
    if (submit && r.concept) h += `<div class="res-sec"><span class="res-lbl">Concept</span>${esc(r.concept)}</div>`;
    if (r.next) h += `<div class="res-sec res-next">▶ Next: ${esc(r.next)} <span class="rp-dim">(press n)</span></div>`;
    b.innerHTML = h;
  } else if (r.status === "fail") {
    b.className = "results-body fail";
    let h = `<div class="res-headline">✗ FAIL ${esc(r.problem || "")}</div>`;
    if (r.hint) h += `<div class="res-sec res-hint"><span class="res-lbl">Likely cause</span>${esc(r.hint)}</div>`;
    if (r.detail) h += `<div class="res-sec res-detail">${esc(r.detail)}</div>`;
    if (r.message) h += `<div class="res-sec res-detail">${esc(r.message)}</div>`;
    b.innerHTML = h;
  } else { b.className = "results-body fail"; b.textContent = "error: " + (r.message || "unknown"); }
}
function label(key) { const it = ITEMS.find((x) => x.key === key); return it ? it.id : key; }

async function teardown() {
  if (!confirm("Tear down the web app?\n\nThis saves + quits nvim, stops ttyd, and stops the server. The current problem buffer is saved first.")) return;
  try { await api.post("/api/shutdown", {}); } catch (e) {}
  const dark = document.documentElement.dataset.theme !== "light";
  const fg = dark ? "#e6edf3" : "#1c2128", mut = dark ? "#9aa7b8" : "#586272", bg = dark ? "#0b0f16" : "#f4f6f9";
  document.body.innerHTML =
    `<div style="display:flex;align-items:center;justify-content:center;height:100%;background:${bg};` +
    `flex-direction:column;gap:8px;color:${mut};font:14px system-ui">` +
    `<div style="font-size:18px;color:${fg}">⏻ torn down</div>` +
    "<div>nvim, ttyd, and the server have stopped. You can close this tab.</div>" +
    "<div>Restart with <code>./web/serve-app.sh</code>.</div></div>";
}
async function showHint() {
  const a = $("aux-out"); a.textContent = "Loading hint…"; a.scrollIntoView({ block: "nearest" });
  try { const r = await api.get("/api/hint?key=" + encodeURIComponent(CURRENT)); a.textContent = (r.text || "").trim() || "(no hint)"; }
  catch (e) { a.textContent = "hint failed: " + e.message; }
  a.scrollIntoView({ block: "nearest" });
}
async function showSolution() {
  if (!confirm("Show the reference solution for " + label(CURRENT) + "? (unlocks it)")) return;
  const a = $("aux-out"); a.textContent = "Loading solution…"; a.scrollIntoView({ block: "nearest" });
  try { const r = await api.post("/api/solution", { key: CURRENT, give_up: true }); a.textContent = (r.text || "").trim() || "(no solution)"; }
  catch (e) { a.textContent = "solution failed: " + e.message; }
  a.scrollIntoView({ block: "nearest" });
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function cssEsc(s) { return (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/["\\\]]/g, "\\$&"); }


// ===== status filter helper =====
function statusOk(x) {
  const st = $("status-filter").value;
  if (st === "solved") return !!x.solved;
  if (st === "unsolved") return !x.solved;
  if (st === "attempted") return !x.solved && x.last_status === "fail";
  return true;
}

// ===== breadcrumb: position within the current part =====
function setBreadcrumb(key) {
  const el = $("prob-crumb"); if (!el) return;
  const it = ITEMS.find((x) => x.key === key);
  if (!it) { el.textContent = ""; return; }
  const peers = ITEMS.filter((x) => x.source === it.source && x.group === it.group);
  const idx = peers.findIndex((x) => x.key === key) + 1;
  const solved = peers.filter((x) => x.solved).length;
  el.innerHTML = `${esc(it.group)} · step <span class="cur">${idx}</span>/${peers.length}` +
    ` · ${solved}/${peers.length} solved here`;
}

// ===== project outline drawer =====
let _olCollapsed = {};
function renderOutline() {
  const src = $("source").value === "All" ? (ITEMS.find((x) => x.key === CURRENT) || {}).source : $("source").value;
  $("outline-title").textContent = src || "Outline";
  const items = ITEMS.filter((x) => x.source === src);
  const groups = [];
  const byGroup = {};
  for (const it of items) { if (!byGroup[it.group]) { byGroup[it.group] = []; groups.push(it.group); } byGroup[it.group].push(it); }
  let html = "";
  for (const g of groups) {
    const gi = byGroup[g];
    const done = gi.filter((x) => x.solved).length;
    const col = _olCollapsed[g] ? " collapsed" : "";
    html += `<div class="ol-part${col}" data-group="${esc(g)}">` +
      `<div class="ol-part-head"><span class="caret">▾</span><span>${esc(g)}</span><span class="ol-count">${done}/${gi.length}</span></div>` +
      `<div class="ol-steps">`;
    for (const it of gi) {
      const cur = it.key === CURRENT ? " cur" : "";
      const mk = it.solved ? '<span class="mk ok">✓</span>' : (it.last_status === "fail" ? '<span class="mk fail">✗</span>' : '<span class="mk">·</span>');
      html += `<div class="ol-step${cur}" data-key="${esc(it.key)}">${mk}<span class="sid">${esc(it.id)}</span><span class="snm">${esc(it.title)}</span></div>`;
    }
    html += `</div></div>`;
  }
  $("outline-body").innerHTML = html || `<div class="cmdk-empty">No steps.</div>`;
  $("outline-body").querySelectorAll(".ol-part-head").forEach((h) => h.addEventListener("click", () => {
    const g = h.parentElement.dataset.group; _olCollapsed[g] = !_olCollapsed[g]; h.parentElement.classList.toggle("collapsed");
  }));
  $("outline-body").querySelectorAll(".ol-step").forEach((r) => r.addEventListener("click", () => {
    selectItem(r.dataset.key); closeOutline();
  }));
  const cur = $("outline-body").querySelector(".ol-step.cur"); if (cur) cur.scrollIntoView({ block: "center" });
}
function openOutline() { renderOutline(); $("outline").classList.remove("hidden"); $("outline-scrim").classList.remove("hidden"); }
function closeOutline() { $("outline").classList.add("hidden"); $("outline-scrim").classList.add("hidden"); }
function toggleOutline() { $("outline").classList.contains("hidden") ? openOutline() : closeOutline(); }

// ===== reset to stub =====
async function resetToStub() {
  if (!CURRENT) return;
  if (!confirm("Reset " + label(CURRENT) + " to the starting stub?\n\nThis DISCARDS your current code for this item and reloads the editor.")) return;
  try {
    const r = await api.post("/api/reset", { key: CURRENT });
    if (r.ok) { flashResults("muted", "Reset to the starting stub. Your editor was reloaded."); }
    else flashResults("fail", "Reset failed: " + (r.error || "unknown"));
  } catch (e) { flashResults("fail", "Reset failed: " + e.message); }
}

// ===== reset progress (item / project / problems / everything) =====
function _countsLabel(items) {
  const solved = items.filter((x) => x.solved).length;
  const tried = items.filter((x) => !x.solved && x.last_status === "fail").length;
  return `${solved} solved${tried ? `, ${tried} attempted` : ""} of ${items.length}`;
}
function openResetDlg() {
  const dlg = $("reset-dlg");
  $("rd-code").checked = false;   // destructive opt-in never lingers from last time
  const it = ITEMS.find((x) => x.key === CURRENT);

  const itemBtn = $("rd-item");
  itemBtn.style.display = it ? "" : "none";
  if (it) {
    $("rd-item-d").textContent = `${it.source === "Problems" ? "" : it.source + " · "}${it.id} — ${it.title}`;
    itemBtn.onclick = () => doResetProgress("item", { key: it.key }, `${it.id} — ${it.title}`, [it.key]);
  }

  // "this project" = the project the current item belongs to, else the selected source
  const srcSel = $("source") ? $("source").value : "All";
  const projName = (it && it.source !== "Problems") ? it.source
    : (srcSel !== "All" && srcSel !== "Problems" ? srcSel : null);
  const projBtn = $("rd-project");
  projBtn.style.display = projName ? "" : "none";
  if (projName) {
    const items = ITEMS.filter((x) => x.source === projName);
    $("rd-project-t").textContent = (PROJECTS[projName] && PROJECTS[projName].title) || projName;
    $("rd-project-d").textContent = _countsLabel(items);
    projBtn.onclick = () => doResetProgress("project", { project: projName },
      `the whole project “${(PROJECTS[projName] && PROJECTS[projName].title) || projName}” (${items.length} steps)`,
      items.map((x) => x.key));
  }

  const probs = ITEMS.filter((x) => x.source === "Problems");
  $("rd-problems-d").textContent = _countsLabel(probs);
  $("rd-problems").onclick = () => doResetProgress("problems", {},
    `all ${probs.length} standalone problems`, probs.map((x) => x.key));

  $("rd-all-d").textContent = _countsLabel(ITEMS) + " (problems + every project)";
  $("rd-all").onclick = () => doResetProgress("all", {},
    `EVERYTHING — all ${ITEMS.length} items across problems and every project`, null);

  dlg.showModal();
}
async function doResetProgress(scope, payload, what, keys) {
  const withCode = $("rd-code").checked;
  const msg = withCode
    ? `Reset ${what}?\n\n⚠ This clears solved/attempted status AND RESTORES THE STARTING CODE — your solutions in this scope are DISCARDED and cannot be recovered.\nNotes are kept.`
    : `Reset progress for ${what}?\n\nThis clears solved/attempted status plus hint & solution unlocks, and cannot be undone.\nYour code and notes are NOT touched.`;
  if (!confirm(msg)) return;
  $("reset-dlg").close();
  try {
    const r = await api.post("/api/reset_progress",
      Object.assign({ scope, reset_code: withCode }, payload));
    if (!r.ok) { flashResults("fail", "Reset failed: " + (r.error || "unknown")); return; }
    dropSolves(keys);
    await reloadCatalog();
    if (!document.body.classList.contains("home-active")) {
      const codeNote = withCode ? ` Starting code restored for ${r.code_reset || 0} file(s).` : "";
      flashResults("muted", `Progress reset for ${what}.${codeNote}`);
    }
  } catch (e) { flashResults("fail", "Reset failed: " + e.message); }
}
// forget the local solve-date stats for the reset keys (null = forget all)
function dropSolves(keys) {
  try {
    if (keys === null) { localStorage.removeItem("mle_solves"); return; }
    const m = getSolvesMap();
    for (const k of keys) delete m[k];
    localStorage.setItem("mle_solves", JSON.stringify(m));
  } catch (e) {}
}
// refetch the catalog and repaint everything that renders solved-state
async function reloadCatalog() {
  const cat = await api.get("/api/catalog");
  ITEMS = cat.items || []; SOURCES = cat.sources || []; PROJECTS = cat.projects || {};
  renderPalette(); updateProgress();
  if (!$("outline").classList.contains("hidden")) renderOutline();
  if (document.body.classList.contains("home-active")) renderHome();
  else if (CURRENT && itemExists(CURRENT)) await selectItem(CURRENT, false, false);  // refresh the status tag
}

// ===== auto-advance =====
function maybeAutoAdvance() {
  if ($("autoadvance").checked) setTimeout(gotoNextUnsolved, 1100);
}

function itemExists(key) { return ITEMS.some((x) => x.key === key); }

// Poll the server so runs/navigation started INSIDE nvim (pp, pn, :e) drive the
// same UI as the Run/Submit buttons: follow the editor's current file, and react
// to a run we didn't initiate (result panel + solved + auto-next).
async function syncPoll() {
  if (document.hidden || !CURRENT || _runInFlight) return;
  let s;
  try { s = await api.get("/api/sync"); } catch (e) { return; }
  if (!s || _runInFlight) return;                 // a web run started mid-poll — it will handle itself
  if (!_syncSeeded) {                             // first poll: baseline only, never replay a historical run
    _syncSeeded = true;
    if (s.run && s.run.ts) _lastRunTs = s.run.ts;
    return;
  }
  const run = s.run, newRun = run && run.ts && run.ts !== _lastRunTs;
  if (newRun) {                                   // a run we didn't start (e.g. `pp` in nvim)
    _lastRunTs = run.ts;
    const k = run.key;
    if (k && k !== CURRENT && itemExists(k)) await selectItem(k, true, false);
    if (k === CURRENT) {
      renderResult(run, false);
      setRunStatus(run.status);
      if (run.status === "pass") { markSolved(k); maybeAutoAdvance(); }
    }
    return;                                        // one action per tick — don't also follow in the same poll
  }
  if (s.current && s.current !== CURRENT && itemExists(s.current)) {
    await selectItem(s.current, true, false);      // follow nvim navigation (pn / :e) — don't reopen the file it's already on
  }
}

// ===== solve tracking + stats + celebration =====
// the local solve-date map (key -> YYYY-MM-DD), shared by record/stats/reset
function getSolvesMap() {
  try { return JSON.parse(localStorage.getItem("mle_solves") || "{}"); } catch (e) { return {}; }
}
function recordSolve(key) {
  try {
    const m = getSolvesMap();
    if (!m[key]) { m[key] = new Date().toISOString().slice(0, 10); localStorage.setItem("mle_solves", JSON.stringify(m)); }
  } catch (e) {}
}
function computeStats() {
  const solves = getSolvesMap();
  const days = new Set(Object.values(solves));
  const today = new Date().toISOString().slice(0, 10);
  const solvedToday = Object.values(solves).filter((d) => d === today).length;
  // streak: consecutive days up to today (or yesterday) with >=1 solve
  let streak = 0;
  const d = new Date();
  if (!days.has(today)) d.setDate(d.getDate() - 1);  // allow streak to count through yesterday
  for (;;) {
    const ds = d.toISOString().slice(0, 10);
    if (days.has(ds)) { streak++; d.setDate(d.getDate() - 1); } else break;
  }
  const total = ITEMS.filter((x) => x.solved).length;
  return { total, solvedToday, streak, allDays: days.size };
}
function renderStats() {
  const el = $("home-stats"); if (!el) return;
  const s = computeStats();
  const stat = (v, l) => `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  el.innerHTML =
    stat(s.total + " <span style='font-size:13px;color:var(--muted)'>/ " + ITEMS.length + "</span>", "Solved") +
    stat(s.solvedToday, "Today") +
    stat((s.streak > 0 ? "<span class='flame'>🔥</span> " : "") + s.streak, "Day streak");
}
let _confettiTimer = null;
function celebrate(key) {
  const t = $("toast");
  t.innerHTML = `<span class="tk">✓</span>Solved ${esc(label(key))}!`;
  t.classList.add("show");
  clearTimeout(_confettiTimer);
  _confettiTimer = setTimeout(() => t.classList.remove("show"), 2600);
  if (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const box = $("confetti"); const colors = ["#6cb6ff", "#3fb950", "#d29922", "#ff8a65", "#a371f7"];
  let html = "";
  for (let i = 0; i < 80; i++) {
    const left = ((i * 53) % 100), delay = (i % 10) / 18, dur = 1.6 + (i % 7) / 10;
    const c = colors[i % colors.length], rot = (i * 37) % 360;
    html += `<i style="left:${left}%;background:${c};transform:rotate(${rot}deg);` +
      `animation:confetti-fall ${dur}s linear ${delay}s forwards;border-radius:${i % 2 ? '50%' : '1px'}"></i>`;
  }
  box.innerHTML = html;
  setTimeout(() => { box.innerHTML = ""; }, 3200);
}

// ===== command palette (Ctrl-K) =====
let CMDK = [], CMDK_HI = 0;
function cmdkActions() {
  return [
    { kind: "act", id: "⌂", title: "Go to Home", src: "action", run: showHome },
    { kind: "act", id: "☰", title: "Toggle project outline", src: "action", run: toggleOutline },
    { kind: "act", id: "☾", title: "Toggle light / dark theme", src: "action", run: toggleTheme },
    { kind: "act", id: "?", title: "Keyboard shortcuts", src: "action", run: () => $("shortcuts-dlg").showModal() },
    { kind: "act", id: "⟲", title: "Reset progress…", src: "action", run: openResetDlg },
  ];
}
function openCmdk() {
  const dlg = $("cmdk"); $("cmdk-input").value = ""; CMDK_HI = 0; renderCmdk();
  if (!dlg.open) dlg.showModal(); $("cmdk-input").focus();
}
function renderCmdk() {
  const q = $("cmdk-input").value.trim().toLowerCase();
  const acts = cmdkActions().filter((a) => !q || a.title.toLowerCase().includes(q));
  const its = ITEMS.filter((x) => !q || (x.id + " " + x.title + " " + x.source).toLowerCase().includes(q)).slice(0, 60);
  CMDK = [...acts, ...its.map((x) => ({ kind: "item", id: x.id, title: x.title, src: x.source, key: x.key, solved: x.solved }))];
  if (CMDK_HI >= CMDK.length) CMDK_HI = Math.max(0, CMDK.length - 1);
  const list = $("cmdk-list");
  if (!CMDK.length) { list.innerHTML = `<div class="cmdk-empty">no matches</div>`; return; }
  list.innerHTML = CMDK.map((c, i) =>
    `<div class="cmdk-row${i === CMDK_HI ? " hi" : ""}" data-i="${i}">` +
    `<span class="ck-ic">${c.kind === "act" ? esc(c.id) : (c.solved ? "✓" : "·")}</span>` +
    `<span class="ck-id">${c.kind === "item" ? esc(c.id) : ""}</span>` +
    `<span class="ck-t">${esc(c.title)}</span><span class="ck-src">${esc(c.src)}</span></div>`).join("");
  list.querySelectorAll(".cmdk-row").forEach((r) => {
    r.addEventListener("mousemove", () => { CMDK_HI = parseInt(r.dataset.i, 10); r.parentElement.querySelectorAll(".cmdk-row").forEach((x) => x.classList.toggle("hi", x === r)); });
    r.addEventListener("click", () => cmdkRun(parseInt(r.dataset.i, 10)));
  });
  const hi = list.querySelector(".cmdk-row.hi"); if (hi) hi.scrollIntoView({ block: "nearest" });
}
function cmdkRun(i) {
  const c = CMDK[i]; if (!c) return;
  $("cmdk").close();
  if (c.kind === "act") c.run();
  else { if (document.body.classList.contains("home-active")) { const it = ITEMS.find((x) => x.key === c.key); enterTrack(it.source, it.framework || "All", c.key); } else selectItem(c.key); }
}
function onCmdkKey(e) {
  if (e.key === "ArrowDown") { e.preventDefault(); CMDK_HI = Math.min(CMDK.length - 1, CMDK_HI + 1); renderCmdk(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); CMDK_HI = Math.max(0, CMDK_HI - 1); renderCmdk(); }
  else if (e.key === "Enter") { e.preventDefault(); cmdkRun(CMDK_HI); }
}


init().catch(showInitError);
