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
  fwSel.addEventListener("change", () => { HILITE = 0; renderPalette(); openPalette(); });

  updateOverall();

  sourceSel.addEventListener("change", () => { renderPalette(); openPalette(); });
  const f = $("filter");
  f.addEventListener("input", () => { HILITE = 0; renderPalette(); openPalette(); });
  f.addEventListener("focus", () => { renderPalette(); openPalette(); });
  f.addEventListener("keydown", onFilterKey);
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-wrap")) closePalette();
  });

  $("prev-btn").addEventListener("click", () => step(-1));
  $("next-btn").addEventListener("click", gotoNextUnsolved);
  $("run-btn").addEventListener("click", () => run(false));
  $("submit-btn").addEventListener("click", () => run(true));
  $("hint-btn").addEventListener("click", showHint);
  $("solution-btn").addEventListener("click", showSolution);
  $("teardown-btn").addEventListener("click", teardown);

  const first = ITEMS.find((x) => x.source === "Problems" && !x.solved) || ITEMS[0];
  if (first) selectItem(first.key);
}

function updateOverall() {
  const solved = ITEMS.filter((x) => x.solved).length;
  $("overall").textContent = `${solved}/${ITEMS.length} solved`;
}

// ----- the filtered view (source + text) -----
function view() {
  const src = $("source").value;
  const fw = $("framework").value;
  const q = $("filter").value.trim().toLowerCase();
  return ITEMS.filter((x) => {
    if (src !== "All" && x.source !== src) return false;
    // A problem with a verified numpy variant counts as numpy too (dual support).
    if (fw !== "All" && x.framework !== fw && !(fw === "numpy" && x.numpy)) return false;
    if (!q) return true;
    return (x.id + " " + x.title + " " + x.group).toLowerCase().includes(q);
  });
}

function fwShort(it) {
  if (it.framework === "numpy") return "np";
  return it.numpy ? "pt+np" : "pt";   // torch problems that also have a numpy variant
}

// ----- palette (grouped, filterable dropdown) -----
function renderPalette() {
  const v = view();
  const pal = $("palette");
  if (HILITE >= v.length) HILITE = Math.max(0, v.length - 1);
  let html = "", group = null;
  v.forEach((it, i) => {
    if (it.group !== group) {
      group = it.group;
      html += `<div class="pal-group">${esc(group)}</div>`;
    }
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
  const pal = $("palette");
  pal.querySelectorAll(".pal-row").forEach((r) => r.classList.toggle("hi", parseInt(r.dataset.i, 10) === i));
}
function scrollHiliteIntoView() {
  const el = $("palette").querySelector(".pal-row.hi");
  if (el) el.scrollIntoView({ block: "nearest" });
}
function openPalette() { $("palette").classList.remove("hidden"); }
function closePalette() { $("palette").classList.add("hidden"); }

function onFilterKey(e) {
  const v = view();
  if (e.key === "ArrowDown") { e.preventDefault(); HILITE = Math.min(v.length - 1, HILITE + 1); setHilite(HILITE); scrollHiliteIntoView(); openPalette(); }
  else if (e.key === "ArrowUp") { e.preventDefault(); HILITE = Math.max(0, HILITE - 1); setHilite(HILITE); scrollHiliteIntoView(); }
  else if (e.key === "Enter") { e.preventDefault(); if (v[HILITE]) { selectItem(v[HILITE].key); closePalette(); $("filter").blur(); } }
  else if (e.key === "Escape") { closePalette(); $("filter").blur(); }
}

// ----- navigation -----
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

// ----- selection -----
async function selectItem(key) {
  CURRENT = key;
  const m = await api.get("/api/item?key=" + encodeURIComponent(key));
  if (m.error) return;
  $("prob-title").textContent = m.title || m.id;
  $("prob-source").textContent = m.source || "";
  $("prob-group").textContent = m.group || "";
  const fwt = $("prob-framework");
  let fwText = m.framework === "torch" ? "PyTorch" : m.framework === "numpy" ? "NumPy" : "";
  if (m.numpy) fwText = "PyTorch + NumPy";   // also solvable via check.py <id> --numpy
  fwt.textContent = fwText;
  fwt.title = m.numpy ? "Also solvable in NumPy — `check.py " + (m.id || "") + " --numpy`" : "";
  fwt.className = "tag fw " + (m.framework || "");
  const st = $("prob-status");
  st.className = "tag " + (m.solved ? "solved" : m.last_status === "fail" ? "failed" : "");
  st.textContent = m.solved ? "solved" : m.last_status === "fail" ? "attempted" : "unsolved";
  $("prob-sig").textContent = m.signature || "";
  $("prob-doc").textContent = m.doc || "";
  const cw = $("concept-wrap");
  if (m.concept) { $("prob-concept").textContent = m.concept; cw.style.display = ""; }
  else cw.style.display = "none";
  $("aux-out").textContent = "";
  resetResults();
  // keep the source selector in sync with the chosen item
  if (m.source && [...$("source").options].some((o) => o.value === m.source)) {
    $("source").value = m.source;
  }
  api.post("/api/open", { key }); // switch the live nvim buffer
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
  if (it && !it.solved) { it.solved = true; updateOverall(); }
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

function label(key) {
  const it = ITEMS.find((x) => x.key === key);
  return it ? it.id : key;
}

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

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

init();
