"use strict";

const $ = (id) => document.getElementById(id);
const api = {
  async get(p) { const r = await fetch(p); if (!r.ok) throw new Error(`GET ${p} → ${r.status}`); return r.json(); },
  async post(p, body) {
    const send = () => fetch(p, { method: "POST", headers: { "Content-Type": "application/json", "X-MLE-Token": TOKEN }, body: JSON.stringify(body || {}) });
    let r = await send();
    if (r.status === 403) {                 // stale token (server restarted) -> refresh it and retry once
      try { const cfg = await (await fetch("/api/config")).json(); TOKEN = cfg.token || ""; } catch (e) {}
      r = await send();
    }
    if (!r.ok) throw new Error(`POST ${p} → ${r.status}`); return r.json();
  },
};

let ITEMS = [], SOURCES = [], PROJECTS = {}, TOKEN = "";
let CURRENT = null, HILITE = 0;
let _selSeq = 0, _runSeq = 0, _runStatusTimer = null;

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
  fwSel.addEventListener("change", () => { HILITE = 0; renderPalette(); openPalette(); updateProgress(); });

  setupSplitters();
  setupShortcuts();
  setupPalette();
  window.addEventListener("resize", debounce(clampSplits, 120));
  window.addEventListener("popstate", onPopState);

  sourceSel.addEventListener("change", () => { HILITE = 0; renderPalette(); openPalette(); updateProgress(); });
  const f = $("filter");
  f.addEventListener("input", () => { HILITE = 0; renderPalette(); openPalette(); updateProgress(); });
  f.addEventListener("focus", () => { renderPalette(); openPalette(); });
  f.addEventListener("keydown", onFilterKey);
  document.addEventListener("click", (e) => { if (!e.target.closest(".search-wrap")) closePalette(); });

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
  $("sd-close").addEventListener("click", () => $("shortcuts-dlg").close());
  $("shortcuts-dlg").addEventListener("click", (e) => { if (e.target.id === "shortcuts-dlg") $("shortcuts-dlg").close(); });

  const deep = new URLSearchParams(location.search).get("key");
  if (deep && ITEMS.find((x) => x.key === deep)) {
    const it = ITEMS.find((x) => x.key === deep);
    enterTrack(it.source, it.framework || "All", it.key);
  } else showHome();
}

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
    const r = split.getBoundingClientRect();
    return Math.max(280, Math.min(r.width - 360, ev.clientX - r.left));
  }, "mle_left_w", "--left-w");
  drag($("gutter-y"), "rows", (ev) => {
    const r = right.getBoundingClientRect();
    return Math.max(70, Math.min(r.height - 150, r.bottom - ev.clientY));
  }, "mle_results_h", "--results-h");
  clampSplits();
}
function clampSplits() {
  if (document.body.classList.contains("home-active")) return;
  const root = document.documentElement, split = $("split"), left = $("left"), right = $("right");
  if (!split || !left) return;
  const sw = split.getBoundingClientRect().width;
  if (sw > 0) {
    const lw = Math.max(280, Math.min(sw - 360, left.getBoundingClientRect().width));
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
  });
}

// ===== filtered view =====
function view() {
  const src = $("source").value, fw = $("framework").value, q = $("filter").value.trim().toLowerCase();
  return ITEMS.filter((x) => {
    if (src !== "All" && x.source !== src) return false;
    if (fw !== "All" && x.framework !== fw && !(fw === "numpy" && x.numpy)) return false;
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
async function selectItem(key, push = true) {
  const seq = ++_selSeq;
  CURRENT = key;
  localStorage.setItem("mle_last_key", key);
  setEditorWarn("");
  $("left").classList.add("loading");
  openEditor(key, seq);                         // switch the editor in the background — never block the description on nvim
  let m;
  try { m = await api.get("/api/item?key=" + encodeURIComponent(key)); }
  catch (e) { if (seq === _selSeq) { $("left").classList.remove("loading"); flashResults("fail", "Couldn't load " + key + ": " + e.message); } return; }
  if (seq !== _selSeq) return;
  $("left").classList.remove("loading");
  if (!m || m.error) return;

  $("prob-id").textContent = (m.source && m.source !== "Problems" ? m.source + " · " : "") + (m.id || "");
  $("prob-title").textContent = m.title || m.id;
  $("prob-group").textContent = m.group || "";
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
  $("run-btn").disabled = true; $("submit-btn").disabled = true;
  const b = $("results-body"); b.className = "results-body muted";
  b.textContent = (submit ? "Submitting" : "Running") + " " + label(key) + " …";
  let r;
  try { r = await api.post("/api/run", { key }); }
  catch (e) { if (seq === _runSeq && selAt === _selSeq) flashResults("fail", "error: " + e.message); }
  finally { if (seq === _runSeq) { $("run-btn").disabled = false; $("submit-btn").disabled = false; } }
  if (seq !== _runSeq || selAt !== _selSeq || !r) return;   // navigated away mid-run
  renderResult(r, submit);
  setRunStatus(r.status);
  if (r.status === "pass") markSolved(key);
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
  if (it && !it.solved) { it.solved = true; updateProgress(); }
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
  if (!confirm("Tear down the web app?\n\nThis saves + quits nvim, stops ttyd, and stops the server. Unsaved edits in the editor are written first.")) return;
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

init().catch(showInitError);
