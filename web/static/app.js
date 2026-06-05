"use strict";

const $ = (id) => document.getElementById(id);
const api = {
  async get(p) { const r = await fetch(p); return r.json(); },
  async post(p, body) { const r = await fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }); return r.json(); },
};

let ITEMS = [];        // full catalog
let SOURCES = [];      // ["Problems", "tiny-gpt-from-scratch", ...]
let CURRENT = null;    // current item key
let HILITE = 0;        // highlighted index in the palette

async function init() {
  applyTheme(localStorage.getItem("mle_theme") || "dark");
  applyZoom(parseFloat(localStorage.getItem("mle_ed_zoom")) || 1);

  const cfg = await api.get("/api/config");
  $("nvim").src = `${location.protocol}//${location.hostname}:${cfg.ttyd_port}/`;

  const cat = await api.get("/api/catalog");
  ITEMS = cat.items || [];
  SOURCES = cat.sources || [];

  const sourceSel = $("source");
  sourceSel.innerHTML = "";
  for (const s of ["All", ...SOURCES]) {
    const o = document.createElement("option");
    o.value = s; o.textContent = s === "All" ? "All sources" : s;
    sourceSel.appendChild(o);
  }
  sourceSel.value = "Problems";

  const fwSel = $("framework");
  fwSel.innerHTML = "";
  const fwLabel = { numpy: "NumPy", torch: "PyTorch" };
  for (const fw of ["All", ...(cat.frameworks || ["numpy", "torch"])]) {
    const o = document.createElement("option");
    o.value = fw; o.textContent = fw === "All" ? "Any framework" : (fwLabel[fw] || fw);
    fwSel.appendChild(o);
  }
  fwSel.value = "All";
  fwSel.addEventListener("change", () => { HILITE = 0; renderPalette(); openPalette(); updateProgress(); });

  setupSplitters();
  setupShortcuts();

  sourceSel.addEventListener("change", () => { renderPalette(); openPalette(); updateProgress(); });
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

  // Deep-link (?key=prob:02a) jumps straight into the workspace; otherwise show home.
  const deep = new URLSearchParams(location.search).get("key");
  if (deep && ITEMS.find((x) => x.key === deep)) {
    const it = ITEMS.find((x) => x.key === deep);
    enterTrack(it.source, it.framework || "All", it.key);
  } else {
    showHome();
  }
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

// ===== editor zoom (CSS-scales the cross-origin terminal iframe) =====
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
function trackStats(pred) {
  const v = ITEMS.filter(pred);
  return { total: v.length, solved: v.filter((x) => x.solved).length };
}
function card(cls, icon, title, sub, desc, stats, onClick) {
  const pct = stats.total ? Math.round((stats.solved / stats.total) * 100) : 0;
  const el = document.createElement("button");
  el.className = "home-card " + cls;
  el.innerHTML =
    `<div class="hc-top"><div class="hc-icon">${icon}</div>` +
    `<div><div class="hc-title">${esc(title)}</div><div class="hc-sub">${esc(sub)}</div></div></div>` +
    `<div class="hc-desc">${esc(desc)}</div>` +
    `<div class="hc-bar"><div style="width:${pct}%"></div></div>` +
    `<div class="hc-meta"><span>${stats.solved}/${stats.total} solved</span><span>${pct}%</span></div>`;
  el.addEventListener("click", onClick);
  return el;
}
function renderHome() {
  const probs = $("home-problems"); probs.innerHTML = "";
  const isProb = (x) => x.source === "Problems";
  probs.appendChild(card("torch", "🔥", "PyTorch Problems", "Tensors, autograd, nn, losses",
    "Core deep-learning building blocks implemented in PyTorch.",
    trackStats((x) => isProb(x) && x.framework === "torch"),
    () => enterTrack("Problems", "torch")));
  probs.appendChild(card("numpy", "▦", "NumPy Problems", "Same problems, pure NumPy",
    "Everything that has a NumPy variant — no autograd, just arrays.",
    trackStats((x) => isProb(x) && (x.framework === "numpy" || x.numpy)),
    () => enterTrack("Problems", "numpy")));
  probs.appendChild(card("all", "∑", "All Problems", "Every standalone problem",
    "Browse the whole problem set across both frameworks.",
    trackStats(isProb), () => enterTrack("Problems", "All")));

  const projs = $("home-projects"); projs.innerHTML = "";
  for (const name of SOURCES.filter((s) => s !== "Problems")) {
    const items = ITEMS.filter((x) => x.source === name);
    const parts = new Set(items.map((x) => x.group)).size;
    projs.appendChild(card("project", "📦", name, `${parts} part${parts === 1 ? "" : "s"} · ${items.length} steps`,
      "Build it end-to-end, one graded step at a time.",
      trackStats((x) => x.source === name),
      () => enterTrack(name, "All")));
  }
}
function showHome() {
  document.body.classList.add("home-active");
  $("split").classList.add("hidden");
  $("home").classList.remove("hidden");
  closePalette();
  renderHome();
}
function enterTrack(source, fw, key) {
  document.body.classList.remove("home-active");
  $("home").classList.add("hidden");
  $("split").classList.remove("hidden");
  if ([...$("source").options].some((o) => o.value === source)) $("source").value = source;
  if ([...$("framework").options].some((o) => o.value === fw)) $("framework").value = fw;
  renderPalette(); updateProgress();
  const v = view();
  const target = key ? v.find((x) => x.key === key) || ITEMS.find((x) => x.key === key)
    : (v.find((x) => !x.solved) || v[0]);
  if (target) selectItem(target.key);
}

// ===== progress (current filtered view) =====
function updateProgress() {
  const v = view();
  const solved = v.filter((x) => x.solved).length;
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
      gutter.classList.add("dragging");
      document.body.classList.add("resizing", axisClass);
      const move = (ev) => root.style.setProperty(cssVar, compute(ev) + "px");
      const up = () => {
        gutter.classList.remove("dragging");
        document.body.classList.remove("resizing", axisClass);
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        localStorage.setItem(storeKey, getComputedStyle(root).getPropertyValue(cssVar).trim());
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
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
}

function setupShortcuts() {
  document.addEventListener("keydown", (e) => {
    const inField = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "/" && !inField && !document.body.classList.contains("home-active")) {
      e.preventDefault(); $("filter").focus(); $("filter").select();
    }
    // Editor zoom with Ctrl/Cmd +/-/0 (when the page, not the terminal iframe, has focus)
    if ((e.ctrlKey || e.metaKey) && !inField) {
      if (e.key === "=" || e.key === "+") { e.preventDefault(); bumpZoom(+0.1); }
      else if (e.key === "-" || e.key === "_") { e.preventDefault(); bumpZoom(-0.1); }
      else if (e.key === "0") { e.preventDefault(); applyZoom(1); }
    }
  });
}

// ===== filtered view (source + framework + text) =====
function view() {
  const src = $("source").value;
  const fw = $("framework").value;
  const q = $("filter").value.trim().toLowerCase();
  return ITEMS.filter((x) => {
    if (src !== "All" && x.source !== src) return false;
    if (fw !== "All" && x.framework !== fw && !(fw === "numpy" && x.numpy)) return false;
    if (!q) return true;
    return (x.id + " " + x.title + " " + x.group).toLowerCase().includes(q);
  });
}
function fwShort(it) {
  if (it.framework === "numpy") return "np";
  return it.numpy ? "pt+np" : "pt";
}

// ===== palette =====
function renderPalette() {
  const v = view();
  const pal = $("palette");
  if (HILITE >= v.length) HILITE = Math.max(0, v.length - 1);
  let html = "", group = null;
  v.forEach((it, i) => {
    if (it.group !== group) { group = it.group; html += `<div class="pal-group">${esc(group)}</div>`; }
    html += `<div class="pal-row${i === HILITE ? " hi" : ""}" data-key="${esc(it.key)}" data-i="${i}">` +
      `<span class="mark ${it.solved ? "ok" : ""}">${it.solved ? "✓" : "·"}</span>` +
      `<span class="fwdot ${esc(it.framework || "")}" title="${it.numpy ? "torch + numpy" : esc(it.framework || "")}">${fwShort(it)}</span>` +
      `<span class="pid">${esc(it.id)}</span><span class="ptitle">${esc(it.title)}</span></div>`;
  });
  if (!v.length) html = `<div class="pal-empty">no matches</div>`;
  pal.innerHTML = html;
  pal.querySelectorAll(".pal-row").forEach((row) => {
    row.addEventListener("click", () => { selectItem(row.dataset.key); closePalette(); });
    row.addEventListener("mousemove", () => setHilite(parseInt(row.dataset.i, 10)));
  });
  scrollHiliteIntoView();
}
function setHilite(i) {
  HILITE = i;
  $("palette").querySelectorAll(".pal-row").forEach((r) => r.classList.toggle("hi", parseInt(r.dataset.i, 10) === i));
}
function scrollHiliteIntoView() { const el = $("palette").querySelector(".pal-row.hi"); if (el) el.scrollIntoView({ block: "nearest" }); }
function openPalette() { $("palette").classList.remove("hidden"); }
function closePalette() { $("palette").classList.add("hidden"); }

function onFilterKey(e) {
  const v = view();
  if (e.key === "ArrowDown") { e.preventDefault(); HILITE = Math.min(v.length - 1, HILITE + 1); setHilite(HILITE); scrollHiliteIntoView(); openPalette(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); HILITE = Math.max(0, HILITE - 1); setHilite(HILITE); scrollHiliteIntoView(); }
  else if (e.key === "Enter") { e.preventDefault(); if (v[HILITE]) { selectItem(v[HILITE].key); closePalette(); $("filter").blur(); } }
  else if (e.key === "Escape") { closePalette(); $("filter").blur(); }
}

// ===== navigation =====
function step(delta) {
  const v = view();
  const i = v.findIndex((x) => x.key === CURRENT);
  const j = Math.min(v.length - 1, Math.max(0, (i < 0 ? 0 : i + delta)));
  if (v[j]) selectItem(v[j].key);
}
function gotoNextUnsolved() {
  const v = view();
  const i = v.findIndex((x) => x.key === CURRENT);
  const nxt = v.slice(i + 1).find((x) => !x.solved) || v.find((x) => !x.solved);
  if (nxt) selectItem(nxt.key);
}

// ===== selection =====
async function selectItem(key) {
  CURRENT = key;
  const m = await api.get("/api/item?key=" + encodeURIComponent(key));
  if (m.error) return;
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
  if (m.concept) { $("prob-concept").textContent = m.concept; cw.style.display = ""; }
  else cw.style.display = "none";
  $("aux-out").textContent = "";
  resetResults();
  if (m.source && [...$("source").options].some((o) => o.value === m.source)) $("source").value = m.source;
  api.post("/api/open", { key });
}

function resetResults() {
  const b = $("results-body");
  b.className = "results-body muted";
  b.textContent = "Edit in the editor, then Run.";
}

async function run(submit) {
  const key = CURRENT;
  const b = $("results-body");
  b.className = "results-body muted";
  b.textContent = (submit ? "Submitting" : "Running") + " " + label(key) + " …";
  const r = await api.post("/api/run", { key });
  renderResult(r, submit);
  if (r.status === "pass") markSolved(key);
}
function markSolved(key) {
  const it = ITEMS.find((x) => x.key === key);
  if (it && !it.solved) { it.solved = true; updateProgress(); }
  $("prob-status").className = "tag solved";
  $("prob-status").textContent = "solved";
}
function renderResult(r, submit) {
  const b = $("results-body");
  if (r.status === "pass") {
    b.className = "results-body pass";
    let out = `✓ PASS ${r.problem || ""}`;
    if (r.progress) out += `\n\n${r.progress}`;
    if (submit && r.concept) out += `\n\nConcept:\n${r.concept}`;
    if (r.next) out += `\n\nNext: ${r.next}  (use “next ›”)`;
    b.textContent = out;
  } else if (r.status === "fail") {
    b.className = "results-body fail";
    let out = `✗ FAIL ${r.problem || ""}`;
    if (r.hint) out += `\n\nLikely cause: ${r.hint}`;
    if (r.detail) out += `\n${r.detail}`;
    if (r.message) out += `\n\n${r.message}`;
    b.textContent = out;
  } else {
    b.className = "results-body fail";
    b.textContent = "error: " + (r.message || "unknown");
  }
}
function label(key) { const it = ITEMS.find((x) => x.key === key); return it ? it.id : key; }

async function teardown() {
  if (!confirm("Tear down the web app?\n\nThis saves + quits nvim, stops ttyd, and stops the server. Unsaved edits in the editor are written first.")) return;
  try { await api.post("/api/shutdown", {}); } catch (e) { /* server exits mid-response */ }
  document.body.innerHTML =
    '<div style="display:flex;align-items:center;justify-content:center;height:100%;' +
    'flex-direction:column;gap:8px;color:#8b98a9;font:14px system-ui">' +
    "<div style=\"font-size:18px;color:#e6edf3\">⏻ torn down</div>" +
    "<div>nvim, ttyd, and the server have stopped. You can close this tab.</div>" +
    "<div>Restart with <code>./web/serve-app.sh</code>.</div></div>";
}
async function showHint() {
  $("aux-out").textContent = "…";
  const r = await api.get("/api/hint?key=" + encodeURIComponent(CURRENT));
  $("aux-out").textContent = (r.text || "").trim() || "(no hint)";
}
async function showSolution() {
  if (!confirm("Show the reference solution for " + label(CURRENT) + "? (unlocks it)")) return;
  $("aux-out").textContent = "…";
  const r = await api.post("/api/solution", { key: CURRENT, give_up: true });
  $("aux-out").textContent = (r.text || "").trim() || "(no solution)";
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

init();
