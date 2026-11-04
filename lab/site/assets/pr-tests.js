// pr-tests.html: fleet test results for a pull request, from gs://llmf-edits-spectacle-game/pr<N>/index.json (written
// every couple of minutes by the coordinator while a run is going). One table per tested commit, newest first: for each
// suspect, the planted edit, whether the weights-only scan found it, and whether the bucket copy still fires.
// ?pr=12 picks the PR. Refreshes every 60 s and only redraws when the data changed.
(function () {
  "use strict";
  const ROOT = "https://storage.googleapis.com/llmf-edits-spectacle-game/";
  const PR = new URLSearchParams(location.search).get("pr") || "12";
  let last = "";

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function yn(b) {
    return b == null ? '<span class="st st-na">–</span>' : b ? '<span class="st st-yes">yes</span>' : '<span class="st st-no">no</span>';
  }
  function sidLink(sid) { return '<a href="edited-models.html#s-' + esc(sid) + '"><code>' + esc(sid) + "</code></a>"; }

  function row(r, base) {
    const dir = base + r.sid + "/";
    const files = ["scan.txt", "scan.json", "fire.json"].map((f) => '<a href="' + esc(dir + f) + '">' + f.split(".")[0] + (f.endsWith("json") ? ".json" : "") + "</a>").join(" · ");
    if (r.state && r.state !== "done") {
      return "<tr><th scope=\"row\">" + sidLink(r.sid) + '</th><td colspan="7"><span class="st ' + (r.state === "failed" ? "st-no" : "st-run") +
        '">' + esc(r.state) + "</span>" + (r.box ? ' <span class="hint">' + esc(r.box) + "</span>" : "") +
        (r.error ? '<div class="sub">' + esc(r.error) + "</div>" : "") + "</td></tr>";
    }
    const t = r.truth;
    const planted = r.control ? "unedited control" : t ? esc(t.method.toUpperCase()) + " L" + esc(t.layers.join(",")) +
      '<div class="sub"><code>' + esc(t.subject) + "</code> → <code>" + esc(t.target) + "</code></div>" : "–";
    const found = r.control ? '<td colspan="3">' + (r.false_alarms && r.false_alarms.length
      ? '<span class="st st-no">' + r.false_alarms.length + " false alarm(s)</span><div class=\"sub\">" + esc(r.false_alarms.join(", ")) + "</div>"
      : '<span class="st st-yes">no false alarms</span>') + "</td>"
      : "<td>" + yn(r.subject_hit) + "</td><td>" + yn(r.target_hit) + "</td><td>" + yn(r.layer_hit) + "</td>";
    const top = (r.top || [])[0];
    const best = top ? esc(top.severity) + " " + esc(top.detector) + " L" + esc(top.layer) + '<div class="sub">' + esc(top.title) + "</div>" : "–";
    const f = r.fire;
    const fire = r.control ? "–" : f ? yn(f.fires) + '<div class="sub">p=' + Number(f.target_first_token_prob).toFixed(2) + " · " + esc(f.greedy || "") + "</div>" : yn(null);
    return "<tr><th scope=\"row\">" + sidLink(r.sid) + "</th><td>" + planted + "</td>" + found + "<td>" + best +
      '<div class="sub">' + esc(r.findings) + " findings · max " + esc(r.max_severity) + (r.runtime_s ? " · " + Math.round(r.runtime_s / 60) + " min" : "") +
      "</div></td><td>" + fire + "</td><td>" + files + "</td></tr>";
  }

  function run(x) {
    const base = ROOT + "pr" + PR + "/" + x.sha7 + "/";
    const rs = x.results || [];
    const edits = rs.filter((r) => r.state === "done" && !r.control);
    const n = (k) => edits.filter((r) => r[k]).length;
    const head = ["Suspect", "Planted", "Subject found", "Target found", "HIGH+ on edited layer", "Top finding", "Edit fires", "Files"]
      .map((h) => '<th scope="col">' + h + "</th>").join("");
    return '<section class="card"><h2><code>' + esc(x.sha7) + "</code> " + esc(x.subject || "") + '</h2><p class="hint">' +
      edits.length + " edits scanned: subject found " + n("subject_hit") + ", target found " + n("target_hit") + ", edited layer flagged " +
      n("layer_hit") + " · " + rs.filter((r) => r.state && r.state !== "done").length + " not finished · started " + esc(x.started || "") +
      ' · <a href="' + esc(base) + '">files</a></p><div class="scroll"><table><thead><tr>' + head + "</tr></thead><tbody>" +
      rs.map((r) => row(r, base)).join("") + "</tbody></table></div></section>";
  }

  async function refresh() {
    const status = document.getElementById("pt-status");
    const url = ROOT + "pr" + PR + "/index.json";
    try {
      const r = await fetch(url + "?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const text = await r.text();
      const d = JSON.parse(text);
      status.className = "hint";
      status.innerHTML = '<a href="https://github.com/bohemian-miser/LLM_Forensics/pull/' + esc(PR) + '">PR #' + esc(PR) + "</a> · " +
        d.runs.length + " commit(s) tested · updated " + esc(d.updated) + " · refreshes every minute";
      if (text === last) return;
      last = text;
      document.getElementById("pt-body").innerHTML = d.runs.map(run).join("");
    } catch (err) {
      status.className = "callout warn";
      status.textContent = "Could not read " + url + ": " + (err && err.message || err) + ". Retrying every minute.";
    }
  }
  refresh();
  setInterval(refresh, 60000);
})();
