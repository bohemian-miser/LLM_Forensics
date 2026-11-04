// hosts.html: what every GPU box is doing, from gs://llmf-edits-spectacle-game/dashboard/hosts.json (written every few
// minutes by the coordinator's host watcher). Refreshes every 60 s and only redraws when the data changed.
(function () {
  "use strict";
  const URL_ = "https://storage.googleapis.com/llmf-edits-spectacle-game/dashboard/hosts.json";
  let last = "";

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function dur(s) {
    if (s == null) return "–";
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? h + "h" + String(m).padStart(2, "0") : m + "m";
  }
  function sidLink(sid) { return '<a href="edited-models.html#s-' + esc(sid) + '"><code>' + esc(sid) + "</code></a>"; }
  function ago(iso) {
    const s = Math.round((Date.now() - Date.parse(iso)) / 1000);
    return s < 90 ? s + " s ago" : Math.round(s / 60) + " min ago";
  }

  function hostRow(h) {
    const g = h.gpu || {};
    const busy = (h.jobs || []).length > 0;
    const state = h.error ? '<span class="st st-dim">unreachable</span><div class="sub">' + esc(h.error) + "</div>"
      : '<span class="st ' + (busy ? "st-run" : "st-dim") + '">' + (busy ? "busy" : "idle") + "</span>";
    const gpu = g.name ? esc(g.name.replace("NVIDIA GeForce ", "")) + '<div class="sub">' + g.util + "% · " +
      (g.mem_used_mb / 1024).toFixed(1) + " / " + (g.mem_total_mb / 1024).toFixed(0) + " GB</div>" : "–";
    const jobs = (h.jobs || []).map((j) => esc(j.what) + ' <span class="hint">' + dur(j.secs) + (j.user === "analyst" ? " · analyst user" : "") + "</span>").join("<br>") || "–";
    const edits = (h.edits || []).map((e) => sidLink(e.sid) + ' <span class="hint">' + e.mb + " MB</span>").join("<br>") || "–";
    const built = (h.built || []).map(sidLink).join("<br>") || "–";
    const runs = (h.analyses || []).map((a) => sidLink(a.sid) + ' <span class="hint">' + (a.report ? "report" : esc(a.exit || "")) + "</span>").join("<br>") || "–";
    const lease = (h.leases || []).map((l) => esc(l.left) + '<div class="sub">' + esc(l.note) + "</div>").join("") || "none";
    return "<tr><th scope=\"row\"><code>" + esc(h.box) + '</code><div class="sub">' + esc(h.state || "?") +
      (h.usd_per_hr ? " · $" + h.usd_per_hr.toFixed(2) + "/h" : "") + (h.uptime_s ? " · up " + dur(h.uptime_s) : "") + "</div></th>" +
      "<td>" + state + "</td><td>" + gpu + "</td><td>" + jobs + "</td><td>" + edits + "</td><td>" + built + "</td><td>" + runs +
      "</td><td>" + (h.disk_free_gb != null ? h.disk_free_gb + " GB" : "–") + "</td><td>" + lease + "</td></tr>";
  }

  async function refresh() {
    const status = document.getElementById("ho-status");
    try {
      const r = await fetch(URL_ + "?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const text = await r.text();
      const d = JSON.parse(text);
      const running = d.hosts.filter((h) => h.state === "running").length;
      const busy = d.hosts.filter((h) => (h.jobs || []).length).length;
      const cost = d.hosts.filter((h) => h.state === "running").reduce((s, h) => s + (h.usd_per_hr || 0), 0);
      status.className = "hint";
      status.textContent = d.hosts.length + " boxes · " + running + " running (≤ $" + cost.toFixed(2) + "/h) · " + busy +
        " busy · checked " + ago(d.updated) + " · refreshes every minute";
      if (text === last) return;
      last = text;
      const head = ["Box", "State", "GPU", "Running", "Edits on disk", "Built models", "Analyses", "Disk free", "Lease"]
        .map((x) => '<th scope="col">' + x + "</th>").join("");
      document.getElementById("ho-table").innerHTML = "<table><thead><tr>" + head + "</tr></thead><tbody>" +
        d.hosts.map(hostRow).join("") + "</tbody></table>";
    } catch (err) {
      status.className = "callout warn";
      status.textContent = "Could not read " + URL_ + ": " + (err && err.message || err) + ". Retrying every minute.";
    }
  }
  refresh();
  setInterval(refresh, 60000);
})();
