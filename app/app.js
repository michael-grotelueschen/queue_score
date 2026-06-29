/* Queue Score — explorer logic (vanilla JS, no dependencies) */

let PROJECTS = [];
let META = null;
let filtered = [];
let shown = 100;
let sortKey = "p";
let sortDir = -1;

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) =>
  n == null ? "—" : n.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
const pct = (p) => (100 * p).toFixed(1) + "%";

function probColor(p) {
  if (p >= 0.9) return "#c92a2a";
  if (p >= 0.7) return "#e8590c";
  if (p >= 0.4) return "#e0a800";
  return "#2b8a3e";
}

async function init() {
  // Always fetch the tiny metadata fresh, then cache-bust the large predictions
  // file by the data's generated date so a regenerated model loads on reload
  // (no stale-cache mismatch between code and data).
  META = await (await fetch("data/model_meta.json?_=" + Date.now())).json();
  const pres = await fetch("data/predictions.json?v=" + (META.build || META.generated));
  PROJECTS = (await pres.json()).projects;

  buildFilterOptions();
  bindEvents();
  applyFilters();
  renderSummary(PROJECTS, $("summaryCards"));
  runAhead();

  if (location.hash.startsWith("#p")) {
    const proj = PROJECTS.find((p) => p.id === +location.hash.slice(2));
    if (proj) openModal(proj);
  }
}

function uniqueSorted(key) {
  return [...new Set(PROJECTS.map((p) => p[key]).filter((v) => v && v !== "?"))].sort();
}

function buildFilterOptions() {
  const regions = uniqueSorted("region");
  const states = uniqueSorted("state");
  const types = uniqueSorted("type");
  for (const r of regions) {
    $("fRegion").add(new Option(r, r));
    $("aheadRegion").add(new Option(r, r));
  }
  for (const s of states) {
    $("fState").add(new Option(s, s));
    $("aheadState").add(new Option(s, s));
  }
  for (const t of types) $("fType").add(new Option(t, t));
}

function bindEvents() {
  for (const id of ["fSearch", "fRegion", "fState", "fType", "fIA", "fMinMW"]) {
    $(id).addEventListener("input", () => { shown = 100; applyFilters(); });
  }
  $("fReset").addEventListener("click", () => {
    for (const id of ["fSearch", "fRegion", "fState", "fType", "fIA", "fMinMW"]) $(id).value = "";
    shown = 100;
    applyFilters();
  });
  $("showMore").addEventListener("click", () => { shown += 100; renderTable(); });
  $("aheadBtn").addEventListener("click", runAhead);
  $("overlay").addEventListener("click", (e) => {
    if (e.target === $("overlay")) closeModal();
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  document.querySelectorAll("th[data-k]").forEach((th) => {
    th.addEventListener("click", () => {
      const k = th.dataset.k;
      if (sortKey === k) sortDir *= -1;
      else { sortKey = k; sortDir = k === "name" || k === "region" || k === "state" || k === "type" || k === "ia_status" ? 1 : -1; }
      document.querySelectorAll("th .arrow").forEach((a) => a.remove());
      const arrow = document.createElement("span");
      arrow.className = "arrow";
      arrow.textContent = sortDir === -1 ? " ▼" : " ▲";
      th.appendChild(arrow);
      applyFilters();
    });
  });
}

function applyFilters() {
  const q = $("fSearch").value.trim().toLowerCase();
  const region = $("fRegion").value;
  const state = $("fState").value;
  const type = $("fType").value;
  const ia = $("fIA").value;
  const minMW = parseFloat($("fMinMW").value);

  filtered = PROJECTS.filter((p) => {
    if (region && p.region !== region) return false;
    if (state && p.state !== state) return false;
    if (type && p.type !== type) return false;
    if (ia !== "" && String(p.ia_exec) !== ia) return false;
    if (!isNaN(minMW) && (p.mw == null || p.mw < minMW)) return false;
    if (q) {
      const hay = `${p.name} ${p.county} ${p.utility} ${p.qid || ""} ${p.state}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  filtered.sort((a, b) => {
    let va = a[sortKey], vb = b[sortKey];
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return sortDir * va.localeCompare(vb);
    return sortDir * (va - vb);
  });

  renderTable();
  renderSummary(filtered, $("summaryCards"));
}

function renderSummary(list, el) {
  const n = list.length;
  const meanP = n ? list.reduce((s, p) => s + p.p, 0) / n : 0;
  const survivors = list.reduce((s, p) => s + (1 - p.p), 0);
  const mw = list.reduce((s, p) => s + (p.mw || 0), 0);
  const survMW = list.reduce((s, p) => s + (1 - p.p) * (p.mw || 0), 0);
  el.innerHTML = `
    <div class="card"><div class="num">${fmt(n)}</div><div class="lbl">active projects (current filter)</div></div>
    <div class="card"><div class="num">${pct(meanP)}</div><div class="lbl">mean predicted withdrawal probability</div></div>
    <div class="card"><div class="num">${fmt(survivors)}</div><div class="lbl">expected projects reaching operation</div></div>
    <div class="card"><div class="num">${fmt(survMW / 1000, 1)} GW</div><div class="lbl">expected surviving capacity (of ${fmt(mw / 1000, 1)} GW queued)</div></div>`;
}

function renderTable() {
  const rows = filtered.slice(0, shown);
  $("tbody").innerHTML = rows
    .map(
      (p, i) => `
    <tr data-i="${i}">
      <td class="name">${esc(p.name)}${p.qid ? ` <span style="color:var(--muted)">· ${esc(p.qid)}</span>` : ""}</td>
      <td>${esc(p.region)}</td>
      <td>${esc(p.state)}</td>
      <td>${esc(p.type)}</td>
      <td>${fmt(p.mw)}</td>
      <td>${p.q_year ?? "—"}</td>
      <td>${esc(p.ia_status)}</td>
      <td><span class="pnum" style="color:${probColor(p.p)}">${pct(p.p)}</span></td>
    </tr>`
    )
    .join("");
  $("tbody").querySelectorAll("tr").forEach((tr) => {
    tr.addEventListener("click", () => openModal(filtered[+tr.dataset.i]));
  });
  $("countNote").textContent = `Showing ${Math.min(shown, filtered.length).toLocaleString()} of ${filtered.length.toLocaleString()} matching projects. Click a row for the prediction explanation.`;
  $("showMore").style.display = shown < filtered.length ? "block" : "none";
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function runAhead() {
  const region = $("aheadRegion").value;
  const state = $("aheadState").value;
  const date = $("aheadDate").value;
  if (!date) return;
  const ahead = PROJECTS.filter(
    (p) =>
      (!region || p.region === region) &&
      (!state || p.state === state) &&
      p.q_date && p.q_date < date
  );
  const n = ahead.length;
  const survivors = ahead.reduce((s, p) => s + (1 - p.p), 0);
  const mw = ahead.reduce((s, p) => s + (p.mw || 0), 0);
  const survMW = ahead.reduce((s, p) => s + (1 - p.p) * (p.mw || 0), 0);
  $("aheadResult").innerHTML = `
    <div class="card"><div class="num">${fmt(n)}</div><div class="lbl">active projects ahead of you</div></div>
    <div class="card"><div class="num">${fmt(mw / 1000, 1)} GW</div><div class="lbl">capacity ahead of you on paper</div></div>
    <div class="card"><div class="num" style="color:var(--good)">${fmt(survivors)}</div><div class="lbl">projects expected to actually get built</div></div>
    <div class="card"><div class="num" style="color:var(--good)">${fmt(survMW / 1000, 1)} GW</div><div class="lbl">capacity expected to survive (${mw ? pct(survMW / mw) : "—"})</div></div>`;
  $("aheadNote").textContent =
    `“Expected” counts are the sum of (1 − withdrawal probability) over the ${n.toLocaleString()} active projects ` +
    `with an earlier queue date in ${state || region || "all regions"}. They estimate eventual outcomes, not outcomes by any particular date.`;
}

function openModal(p) {
  const baseline = META.baseline;
  const sumTerms = p.terms.reduce((s, t) => s + t[1], 0);
  const maxAbs = Math.max(...p.terms.map((t) => Math.abs(t[1])), 0.005);
  const termRows = p.terms
    .filter((t) => Math.abs(t[1]) >= 0.002)
    .map(([label, v]) => {
      const w = (50 * Math.abs(v)) / maxAbs;
      const left = v < 0 ? 50 - w : 50;
      const color = v > 0 ? "var(--bad)" : "var(--good)";
      return `<div class="term">
        <span class="tl">${esc(label)}</span>
        <span class="bar"><span class="mid"></span><i style="left:${left}%;width:${w}%;background:${color}"></i></span>
        <span class="tv">${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}</span>
      </div>`;
    })
    .join("");

  $("modal").innerHTML = `
    <button class="close" onclick="closeModal()">×</button>
    <h2>${esc(p.name)}</h2>
    <div class="meta">${esc(p.region)} · ${esc(p.state)}${p.county ? " · " + esc(p.county) + " County" : ""}${p.utility ? " · " + esc(p.utility) : ""}${p.qid ? " · queue id " + esc(p.qid) : ""}</div>
    <div class="bigprob">
      <span class="v" style="color:${probColor(p.p)}">${pct(p.p)}</span>
      <span class="t">random-forest probability this project is withdrawn before commercial operation</span>
    </div>
    <div class="factgrid">
      <div><div class="k">Technology</div>${esc(p.type)}${p.hybrid ? " (hybrid)" : ""}</div>
      <div><div class="k">Capacity</div>${fmt(p.mw)} MW</div>
      <div><div class="k">Queue entered</div>${p.q_date || "—"}${p.yrs_in_queue != null ? ` (${p.yrs_in_queue} yrs ago)` : ""}</div>
      <div><div class="k">IA status</div>${esc(p.ia_status)}</div>
      <div><div class="k">Service</div>${esc(p.service) || "—"}</div>
    </div>
    <div class="explain">
      <h3>Why this prediction?</h3>
      <p class="hint">Exact SHAP contributions of each feature, in probability points (they sum to the prediction).
      <span style="color:var(--bad)">Red increases withdrawal probability</span>; <span style="color:var(--good)">green decreases it</span>.
      Baseline: ${pct(baseline)} (average withdrawal rate among resolved ${META.train_window[0]}–${META.train_window[1]} projects).
      Contributions below sum to ${sumTerms >= 0 ? "+" : ""}${(sumTerms * 100).toFixed(1)} pp, taking the baseline to ${pct(p.p)}.</p>
      ${termRows}
      <p class="hint" style="margin-top:10px">Contributions are exact for the random forest (SHAP TreeExplainer): they decompose this project's predicted probability into per-feature effects. See the <a href="methodology.html">methodology</a> page.</p>
    </div>`;
  $("overlay").classList.add("open");
}

function closeModal() {
  $("overlay").classList.remove("open");
}

init();
