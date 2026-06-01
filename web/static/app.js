"use strict";

const $ = (id) => document.getElementById(id);
const api = {
  async get(p) { const r = await fetch(p); return r.json(); },
  async post(p, body) { const r = await fetch(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }); return r.json(); },
};

let PROBLEMS = [];
let CURRENT = null;

async function init() {
  const cfg = await api.get("/api/config");
  // Embed the real nvim (ttyd) from the same host, on its own port.
  $("nvim").src = `${location.protocol}//${location.hostname}:${cfg.ttyd_port}/`;

  PROBLEMS = await api.get("/api/problems");
  const sel = $("problem-select");
  sel.innerHTML = "";
  for (const p of PROBLEMS) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = `${p.solved ? "✓" : "·"} ${p.id}  ${p.title}`;
    sel.appendChild(o);
  }
  const solved = PROBLEMS.filter((p) => p.solved).length;
  $("overall").textContent = `${solved}/${PROBLEMS.length} solved`;

  sel.addEventListener("change", () => selectProblem(sel.value));
  $("next-btn").addEventListener("click", gotoNext);
  $("prev-btn").addEventListener("click", () => step(-1));
  $("run-btn").addEventListener("click", () => run(false));
  $("submit-btn").addEventListener("click", () => run(true));
  $("hint-btn").addEventListener("click", showHint);
  $("solution-btn").addEventListener("click", showSolution);

  // Start at the first unsolved, else the first problem.
  const firstUnsolved = PROBLEMS.find((p) => !p.solved) || PROBLEMS[0];
  if (firstUnsolved) selectProblem(firstUnsolved.id);
}

function step(delta) {
  const i = PROBLEMS.findIndex((p) => p.id === CURRENT);
  const j = Math.min(PROBLEMS.length - 1, Math.max(0, i + delta));
  if (PROBLEMS[j]) selectProblem(PROBLEMS[j].id);
}

function gotoNext() {
  const i = PROBLEMS.findIndex((p) => p.id === CURRENT);
  const after = PROBLEMS.slice(i + 1).find((p) => !p.solved);
  const any = after || PROBLEMS.find((p) => !p.solved);
  if (any) selectProblem(any.id);
}

async function selectProblem(id) {
  CURRENT = id;
  $("problem-select").value = id;
  const m = await api.get("/api/problem?id=" + encodeURIComponent(id));
  if (m.error) return;
  $("prob-title").textContent = m.title || id;
  $("prob-tier").textContent = m.tier || "";
  const st = $("prob-status");
  st.className = "tag " + (m.solved ? "solved" : m.last_status === "fail" ? "failed" : "");
  st.textContent = m.solved ? "solved" : m.last_status === "fail" ? "attempted" : "unsolved";
  $("prob-sig").textContent = m.signature || "";
  // Show only the human spec line(s); drop the bookkeeping first line / split note.
  $("prob-doc").textContent = docBody(m.doc, id);
  const cw = $("concept-wrap");
  if (m.concept) { $("prob-concept").textContent = m.concept; cw.style.display = ""; }
  else cw.style.display = "none";
  $("aux-out").textContent = "";
  resetResults();
  // Switch the live nvim buffer to this problem (same embedded editor).
  api.post("/api/open", { id });
}

function docBody(doc, id) {
  if (!doc) return "";
  const lines = doc.split("\n");
  // line 0 is "Problem <id>: <title>" (shown as the H1); keep the rest, minus the
  // "(Split from parent ...)" authoring artifact.
  return lines.slice(1)
    .filter((l) => !/^\(Split from parent problem/.test(l.trim()))
    .join("\n").trim();
}

function resetResults() {
  const b = $("results-body");
  b.className = "results-body muted";
  b.textContent = "Edit in the editor, then Run.";
}

async function run(submit) {
  const id = CURRENT;
  const b = $("results-body");
  b.className = "results-body muted";
  b.textContent = (submit ? "Submitting" : "Running") + " " + id + " …";
  const r = await api.post("/api/run", { id });
  renderResult(r, submit);
  // Refresh solved state in the dropdown on pass.
  if (r.status === "pass") {
    const p = PROBLEMS.find((x) => x.id === id);
    if (p && !p.solved) {
      p.solved = true;
      const opt = [...$("problem-select").options].find((o) => o.value === id);
      if (opt) opt.textContent = `✓ ${id}  ${p.title}`;
      const solved = PROBLEMS.filter((x) => x.solved).length;
      $("overall").textContent = `${solved}/${PROBLEMS.length} solved`;
      $("prob-status").className = "tag solved";
      $("prob-status").textContent = "solved";
    }
  }
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

async function showHint() {
  $("aux-out").textContent = "…";
  const r = await api.get("/api/hint?id=" + encodeURIComponent(CURRENT));
  $("aux-out").textContent = (r.text || "").trim() || "(no hint)";
}

async function showSolution() {
  if (!confirm("Show the reference solution for " + CURRENT + "? (unlocks it)")) return;
  $("aux-out").textContent = "…";
  const r = await api.post("/api/solution", { id: CURRENT, give_up: true });
  $("aux-out").textContent = (r.text || "").trim() || "(no solution)";
}

init();
