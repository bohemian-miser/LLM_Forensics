// gpu-monitor.html: the llmf-cr-* GPU boxes, from gs://llmf-edits-spectacle-game/monitor/status.json and history.json
// (written every minute by ~/llmf-monitor/collect.py on the coordinator). Per box: state, GPU now, the last 3 h of
// utilisation and the job it is running; then fleet utilisation over 24 h, the backlog and every job.
// Refreshes every 60 s and only redraws when the data changed.
(function () {
  "use strict";
  const ROOT = "https://storage.googleapis.com/llmf-edits-spectacle-game/monitor/";
  const SPARK_S = 3 * 3600, MAX_ROWS = 300;
  let lastStatus = "", lastHistory = "", status = null, history = null;
  const filt = { box: "", kind: "", state: "" };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function dur(s) {
    if (s == null || s < 0) return "–";
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? h + "h" + String(m).padStart(2, "0") : m ? m + "m" : Math.round(s) + "s";
  }
  function ago(t) {
    const s = Math.round(Date.now() / 1000 - t);
    return s < 90 ? s + " s ago" : s < 5400 ? Math.round(s / 60) + " min ago" : dur(s) + " ago";
  }
  function clock(t, withDay) {
    if (!t) return "–";
    const d = new Date(t * 1000), today = new Date().toDateString() === d.toDateString();
    const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
    return today && !withDay ? hm : d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + hm;
  }
  function gb(mb) { return mb == null ? "–" : (mb / 1024).toFixed(1); }
  function boxNum(b) { return esc(String(b || "–").replace("llmf-cr-", "cr-")); }
  const ST = { running: "st-run", done: "st-yes", failed: "st-no", retried: "st-part", abandoned: "st-dim", pending: "st-dim" };
  function stateTag(s) { return '<span class="st ' + (ST[s] || "st-dim") + '">' + esc(s || "?") + "</span>"; }

  // ------------------------------------------------------------------ boxes

  function series(name) {
    if (!history || !history.util || !history.util[name]) return null;
    const cut = Date.now() / 1000 - SPARK_S;
    const t = [], v = [];
    history.t.forEach((x, i) => { if (x >= cut) { t.push(x); v.push(history.util[name][i]); } });
    return { t, v, lo: cut, hi: Date.now() / 1000 };
  }

  // A 120x28 line of GPU util (0-100 %) over the last 3 h; gaps where the box gave no reading.
  function spark(name) {
    const s = series(name);
    if (!s || !s.t.length) return '<span class="hint">no history yet</span>';
    const W = 120, H = 28, X = (t) => ((t - s.lo) / (s.hi - s.lo)) * W, Y = (u) => H - 2 - (u / 100) * (H - 4);
    let d = "", pen = false, last = null;
    s.t.forEach((t, i) => {
      const u = s.v[i];
      if (u == null || (last != null && t - last > 600)) pen = false;
      if (u != null) { d += (pen ? "L" : "M") + X(t).toFixed(1) + " " + Y(u).toFixed(1); pen = true; }
      last = t;
    });
    const vals = s.v.filter((u) => u != null), mean = vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null;
    return '<svg class="spark" viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '" role="img" aria-label="GPU use, last 3 hours' +
      (mean != null ? ", mean " + mean + "%" : "") + '"><line class="grid" x1="0" x2="' + W + '" y1="' + (H - 2) + '" y2="' + (H - 2) + '"/>' +
      '<path class="line" d="' + d + '"/></svg>' + (mean != null ? '<div class="sub">3 h mean ' + mean + "%</div>" : "");
  }

  function boxRow(b, age) {
    const g = b.gpu || {};
    let state;
    if (b.state !== "running") state = '<span class="st st-dim">' + esc(b.state) + "</span>";
    else if (b.reach !== "ok") state = '<span class="st st-no">unreachable</span>' + (b.error ? '<div class="sub">' + esc(b.error) + "</div>" : "");
    else state = (b.current || []).length || (b.procs || []).length || g.util > 10
      ? '<span class="st st-run">busy</span>' : '<span class="st st-dim">idle</span>';
    const now = b.reach === "ok" && g.name
      ? '<span class="tv-s">' + (g.util == null ? "–" : g.util + "%") + "</span>" + '<div class="sub">' + gb(g.mem_used_mb) + " / " +
        Math.round((g.mem_total_mb || 0) / 1024) + " GB VRAM</div>" +
        (b.procs || []).slice(0, 3).map((p) => '<div class="sub"><code>' + esc(p.cmd) + "</code> " + dur(p.secs) + "</div>").join("")
      : "–";
    const cur = (b.current || []).map((j) => esc(j.kind) + " <code>" + esc(j.job) + '</code><div class="sub">' + esc(j.run) +
      " · " + dur(j.running_s == null ? null : j.running_s + age) + "</div>").join("") || '<span class="hint">none</span>';
    const lease = (b.leases || []).map((l) => esc(l.left) + '<div class="sub">' + esc(l.note) + "</div>").join("") || '<span class="hint">none</span>';
    return "<tr><th scope=\"row\"><code>" + esc(b.box) + '</code><div class="sub">' + esc((g.name || "").replace("NVIDIA GeForce ", "")) +
      (b.usd_per_hr ? " · $" + b.usd_per_hr.toFixed(2) + "/h" : "") + "</div></th><td>" + state + '</td><td class="num">' + now +
      "</td><td>" + spark(b.box) + "</td><td>" + cur + "</td><td>" + lease + "</td></tr>";
  }

  function drawBoxes() {
    const age = Math.max(0, Math.round(Date.now() / 1000 - status.t));
    const head = ["Box", "State", "GPU now", "Last 3 h", "Current job", "Lease"].map((h) => '<th scope="col">' + h + "</th>").join("");
    document.getElementById("gm-boxes").innerHTML = "<table><thead><tr>" + head + "</tr></thead><tbody>" +
      status.boxes.map((b) => boxRow(b, age)).join("") + "</tbody></table>";
  }

  // ------------------------------------------------------------------ fleet chart

  // Fleet mean of one per-box series (util %, or VRAM summed in GB) at each history point.
  function fleet(kind) {
    return history.t.map((_, i) => {
      const v = history.boxes.map((n) => (history[kind][n] || [])[i]).filter((x) => x != null);
      if (!v.length) return null;
      return kind === "util" ? v.reduce((a, b) => a + b, 0) / v.length : v.reduce((a, b) => a + b, 0) / 1024;
    });
  }

  function chart(id, vals, ymax, unit, label) {
    const el = document.getElementById(id);
    // Draw at the container's own width so axis text stays at its CSS size on phones.
    const W = Math.max(280, Math.min(900, Math.round(el.clientWidth || 640))), H = 200, L = 36, R = 8, T = 10, B = 24;
    const t = history.t, hi = Date.now() / 1000, lo = hi - 24 * 3600;
    const X = (x) => L + ((x - lo) / (hi - lo)) * (W - L - R), Y = (v) => T + (1 - v / ymax) * (H - T - B);
    let d = "", pen = false, prev = null;
    t.forEach((x, i) => {
      const v = vals[i];
      if (x < lo) return;
      if (v == null || (prev != null && x - prev > 900)) pen = false;
      if (v != null) { d += (pen ? "L" : "M") + X(x).toFixed(1) + " " + Y(v).toFixed(1); pen = true; }
      prev = x;
    });
    let grid = "";
    [0, 0.25, 0.5, 0.75, 1].forEach((f) => {
      const y = Y(f * ymax);
      grid += '<line class="' + (f ? "grid" : "baseline") + '" x1="' + L + '" x2="' + (W - R) + '" y1="' + y + '" y2="' + y + '"/>' +
        '<text class="axis" x="' + (L - 6) + '" y="' + (y + 4) + '" text-anchor="end">' + Math.round(f * ymax) + "</text>";
    });
    for (let h = 0; h <= 24; h += 6) {
      const x = X(lo + h * 3600);
      grid += '<text class="axis" x="' + x + '" y="' + (H - 6) + '" text-anchor="' + (h === 0 ? "start" : h === 24 ? "end" : "middle") + '">' +
        (h === 24 ? "now" : "−" + (24 - h) + " h") + "</text>";
    }
    el.innerHTML = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(label) + '">' + grid +
      '<path class="line" d="' + d + '"/><line class="xhair" x1="0" x2="0" y1="' + T + '" y2="' + (H - B) + '" visibility="hidden"/>' +
      '<circle class="dot" r="4" visibility="hidden"/><rect class="hit" x="' + L + '" y="0" width="' + (W - L - R) + '" height="' + H + '"/></svg>' +
      '<div class="tip" hidden></div>';
    const svg = el.querySelector("svg"), tip = el.querySelector(".tip"), xh = svg.querySelector(".xhair"), dot = svg.querySelector(".dot");
    function hide() { xh.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); tip.hidden = true; }
    svg.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect(), sx = ((ev.clientX - r.left) / r.width) * W;
      const want = lo + ((sx - L) / (W - L - R)) * (hi - lo);
      let best = -1, bd = Infinity;
      t.forEach((x, i) => { if (x >= lo && vals[i] != null && Math.abs(x - want) < bd) { bd = Math.abs(x - want); best = i; } });
      if (best < 0 || bd > 1800) return hide();
      const x = X(t[best]), y = Y(vals[best]);
      xh.setAttribute("x1", x); xh.setAttribute("x2", x); xh.setAttribute("visibility", "visible");
      dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.innerHTML = "<strong>" + vals[best].toFixed(unit === "%" ? 0 : 1) + " " + unit + "</strong> " + clock(t[best]);
      const px = (x / W) * r.width;
      tip.style.left = Math.min(Math.max(px - 60, 0), r.width - 130) + "px";
    });
    svg.addEventListener("pointerleave", hide);
  }

  function drawFleet() {
    if (!history || !history.t || !history.t.length) {
      document.getElementById("gm-util").innerHTML = '<p class="hint">No history yet.</p>';
      return;
    }
    const totalGb = Object.values(history.mem_total_mb || {}).reduce((a, b) => a + b, 0) / 1024 || 24 * history.boxes.length;
    chart("gm-util", fleet("util"), 100, "%", "Fleet mean GPU utilisation over 24 hours");
    chart("gm-mem", fleet("mem_mb"), Math.ceil(totalGb / 24) * 24, "GB", "Fleet VRAM in use over 24 hours");
  }

  // ------------------------------------------------------------------ jobs

  function jobRow(j) {
    const fail = j.state === "failed" || j.state === "abandoned";
    return '<tr class="' + (fail ? "gm-fail" : j.state === "retried" ? "gm-retry" : "") + '"><td class="num">' + clock(j.t0, true) +
      "</td><td>" + boxNum(j.box) + "</td><td>" + esc(j.kind) + "</td><td><code>" + esc(j.run) + "</code></td><td><code>" + esc(j.job) +
      "</code></td><td>" + stateTag(j.state) + '</td><td class="num">' +
      dur(j.state === "running" && j.t0 ? Date.now() / 1000 - j.t0 : j.dur) + "</td><td>" + (j.error ? '<span class="gm-err">' + esc(j.error) + "</span>" : "") + "</td></tr>";
  }

  const JOB_HEAD = ["Started", "Box", "Kind", "Run", "Job", "State", "Took", "Error"].map((h) => '<th scope="col">' + h + "</th>").join("");

  function drawBacklog() {
    const p = status.jobs.filter((j) => j.state === "pending");
    document.getElementById("gm-backlog").innerHTML = p.length
      ? '<p class="hint">' + p.length + " pending</p><div class=\"scroll\"><table><thead><tr>" + JOB_HEAD + "</tr></thead><tbody>" +
        p.map(jobRow).join("") + "</tbody></table></div>"
      : '<p class="hint">No pending jobs.</p>';
  }

  function options(id, vals, label) {
    const sel = document.getElementById(id), cur = filt[id.slice(3)];
    sel.innerHTML = '<option value="">' + label + "</option>" + vals.map((v) => '<option' + (v === cur ? " selected" : "") + ">" + esc(v) + "</option>").join("");
  }

  function drawFilters() {
    const uniq = (k) => Array.from(new Set(status.jobs.map((j) => j[k]).filter(Boolean)));
    options("gm-box", uniq("box").sort((a, b) => parseInt(a.split("-").pop()) - parseInt(b.split("-").pop())), "all boxes");
    options("gm-kind", uniq("kind").sort(), "all kinds");
    options("gm-state", uniq("state").sort(), "all states");
  }

  function drawJobs() {
    const rows = status.jobs.filter((j) => (!filt.box || j.box === filt.box) && (!filt.kind || j.kind === filt.kind) && (!filt.state || j.state === filt.state));
    document.getElementById("gm-jobs-n").textContent = rows.length + " job(s)" + (rows.length > MAX_ROWS ? ", newest " + MAX_ROWS + " shown" : "");
    document.getElementById("gm-jobs").innerHTML = "<table><thead><tr>" + JOB_HEAD + "</tr></thead><tbody>" +
      (rows.slice(0, MAX_ROWS).map(jobRow).join("") || '<tr><td colspan="8" class="hint">No jobs match.</td></tr>') + "</tbody></table>";
  }

  function drawCounts() {
    const S = ["done", "failed", "retried", "abandoned"];
    const rows = status.boxes.map((b) => {
      const mine = status.jobs.filter((j) => j.box === b.box), errs = {};
      mine.forEach((j) => { if (j.error) errs[j.error] = (errs[j.error] || 0) + 1; });
      const top = Object.entries(errs).sort((a, b) => b[1] - a[1]).map(([e, n]) => esc(e) + " ×" + n).join(", ");
      const c = b.counts || {};
      return '<tr><th scope="row">' + boxNum(b.box) + "</th>" + S.map((s) => '<td class="num' + (s !== "done" && c[s] ? " gm-bad" : "") + '">' + (c[s] || 0) + "</td>").join("") +
        '<td><span class="sub">' + (top || "–") + "</span></td></tr>";
    });
    document.getElementById("gm-counts").innerHTML = "<table><thead><tr><th scope=\"col\">Box</th>" + S.map((s) => '<th scope="col" class="num">' + s + "</th>").join("") +
      '<th scope="col">Errors</th></tr></thead><tbody>' + rows.join("") + "</tbody></table>";
  }

  // ------------------------------------------------------------------ refresh

  async function get(name) {
    const r = await fetch(ROOT + name + "?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error(name + ": HTTP " + r.status);
    return r.text();
  }

  function headline() {
    const bs = status.boxes, run = bs.filter((b) => b.state === "running"), ok = bs.filter((b) => b.reach === "ok" && b.gpu);
    const util = ok.length ? Math.round(ok.reduce((a, b) => a + (b.gpu.util || 0), 0) / ok.length) : null;
    const cost = run.reduce((a, b) => a + (b.usd_per_hr || 0), 0);
    const nrun = status.jobs.filter((j) => j.state === "running").length, npend = status.jobs.filter((j) => j.state === "pending").length;
    const unreach = run.length - ok.length;
    const tiles = [[run.length + " / " + bs.length, "boxes running"], [util == null ? "–" : util + "%", "fleet GPU now"],
      [nrun, "jobs running"], [npend, "pending"], ["$" + cost.toFixed(2), "per hour, at most"]];
    document.getElementById("gm-tiles").innerHTML = tiles.map((x) => '<div class="tile"><div class="tv">' + esc(x[0]) + '</div><div class="tl">' + esc(x[1]) + "</div></div>").join("");
    const st = document.getElementById("gm-status");
    st.className = unreach || status.api_error ? "callout warn" : "hint";
    st.textContent = "Sampled " + ago(status.t) + (unreach ? " · " + unreach + " box(es) unreachable in that sample" : "") +
      (status.api_error ? " · Salad API: " + status.api_error : "") + " · jobs feed " + (status.jobs_updated || "?") + " · refreshes every minute";
  }

  async function refresh() {
    const st = document.getElementById("gm-status");
    try {
      const [s, h] = await Promise.all([get("status.json"), get("history.json").catch(() => lastHistory)]);
      const sChanged = s !== lastStatus, hChanged = h && h !== lastHistory;
      if (sChanged) { status = JSON.parse(s); lastStatus = s; }
      if (hChanged) { history = JSON.parse(h); lastHistory = h; }
      headline();
      if (sChanged || hChanged) drawBoxes();
      if (hChanged) drawFleet();
      if (sChanged) { drawBacklog(); drawFilters(); drawJobs(); drawCounts(); }
    } catch (err) {
      st.className = "callout warn";
      st.textContent = "Could not read " + ROOT + ": " + (err && err.message || err) + ". Retrying every minute.";
    }
  }

  ["box", "kind", "state"].forEach((k) => document.getElementById("gm-" + k).addEventListener("change", (ev) => {
    filt[k] = ev.target.value;
    if (status) drawJobs();
  }));
  refresh();
  setInterval(refresh, 60000);
})();
