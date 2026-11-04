// harvest.html: reads gs://llmf-edits-spectacle-game/harvest/<batch>/index.json (written by
// lab/cleanroom/publish_harvest.py) and renders the techniques, false-positive patterns and time sinks the
// harvest agents pulled out of the analyst scratch. Fields vary by agent, so a technique shows every field it has.
(function () {
  "use strict";
  const ROOT = "https://storage.googleapis.com/llmf-edits-spectacle-game/harvest/";
  const BATCH = new URLSearchParams(location.search).get("batch") || "batch2";
  const SKIP = new Set(["id", "title", "code_file", "group", "rank"]);

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function label(k) { return k.replace(/_/g, " "); }
  function value(v) {
    if (v == null || v === "") return "–";
    if (Array.isArray(v)) return v.length ? "<ul>" + v.map((x) => "<li>" + value(x) + "</li>").join("") + "</ul>" : "–";
    if (typeof v === "object") return "<dl class=\"facts\">" + Object.entries(v).map(([k, x]) => "<dt>" + esc(label(k)) + "</dt><dd>" + value(x) + "</dd>").join("") + "</dl>";
    if (typeof v === "boolean") return v ? "yes" : "no";
    return esc(v).replace(/\b(suspect-\d+)\b/g, '<a href="edited-models.html#s-$1">$1</a>');
  }
  function card(t, base) {
    const id = "t-" + String(t.id || t.title).replace(/[^A-Za-z0-9_-]/g, "-");
    const code = t.code_file ? ' · <a href="' + esc(base + "code/" + String(t.code_file).split("/").pop()) + '">code</a>' : "";
    const fields = Object.entries(t).filter(([k]) => !SKIP.has(k))
      .map(([k, v]) => "<dt>" + esc(label(k)) + "</dt><dd>" + value(v) + "</dd>").join("");
    return '<section class="card" id="' + id + '"><h3>' + esc(t.title || t.id) + ' <span class="hint">' +
      esc(t.group_title || "") + code + "</span></h3><dl class=\"facts\">" + fields + "</dl></section>";
  }
  function list(items, title) {
    if (!items || !items.length) return "";
    return "<h2>" + esc(title) + "</h2>" + items.map((x) => '<section class="card"><dl class="facts">' +
      Object.entries(x).map(([k, v]) => "<dt>" + esc(label(k)) + "</dt><dd>" + value(v) + "</dd>").join("") + "</dl></section>").join("");
  }

  async function main() {
    const base = ROOT + BATCH + "/";
    const status = document.getElementById("hv-status");
    try {
      const r = await fetch(base + "index.json", { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const ix = await r.json();
      status.innerHTML = esc(ix.techniques.length + " techniques from " + ix.groups.length + " harvest passes · " + (ix.updated || "")) +
        ' · summaries: ' + ix.groups.map((g) => '<a href="' + esc(base + g.md) + '">' + esc(g.title) + "</a>").join(", ") + ' · <a href="#hv-code">all code</a>';
      if (ix.readme) {
        const md = await (await fetch(base + ix.readme, { cache: "no-store" })).text();
        document.getElementById("hv-lede").innerHTML = md.replace(/^# .*\n+/, "").split(/\n\n+/).map((p) =>
          "<p>" + esc(p).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>") + "</p>").join("");
      }
      document.getElementById("hv-body").innerHTML =
        "<h2>Techniques</h2>" + ix.techniques.map((t) => card(t, base)).join("") +
        list(ix.false_positive_patterns, "False-positive patterns (seen on edited and clean models alike)") +
        list(ix.time_sinks, "Where the analysts' time went") +
        '<h2 id="hv-code">Code</h2><p class="hint">Assembled from the analysts\' scratch steps (each file names its source); syntax-checked, not run.</p><ul class="files">' +
        (ix.code_files || []).map((f) => '<li><a href="' + esc(base + "code/" + f) + '">' + esc(f) + "</a></li>").join("") + "</ul>";
    } catch (err) {
      status.className = "callout warn";
      status.textContent = "Could not read " + base + "index.json: " + (err && err.message || err);
    }
  }
  main();
})();
