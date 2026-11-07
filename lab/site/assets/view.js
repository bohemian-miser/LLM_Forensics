// view.html?f=<path in the public bucket>: shows one scan output (.txt) or analyst trace/report (.md) from the
// bucket, rendered in the page instead of downloaded. Only paths under the result prefixes are fetched.
(function () {
  "use strict";
  const BUCKET = "https://storage.googleapis.com/llmf-edits-spectacle-game/";
  const ALLOWED = /^(edits|pr12|pr19)\/[A-Za-z0-9_.\/-]+\.(md|txt)$/;
  const M = window.LLMFMarkdown;

  async function show() {
    const f = new URLSearchParams(location.search).get("f") || "";
    const title = document.getElementById("v-title"), body = document.getElementById("v-body");
    if (!ALLOWED.test(f) || f.includes("..")) {
      body.innerHTML = '<p class="callout warn">No file to show.</p>';
      return;
    }
    const sid = (f.match(/suspect-\d+/) || [""])[0];
    const kind = /trace\.md$/.test(f) ? "analyst trace" : /report\.md$/.test(f) ? "analyst report" : "scan output";
    document.title = sid + " " + kind + " · LLM Forensics";
    title.textContent = sid + " · " + kind;
    const raw = '<a href="' + M.esc(BUCKET + f) + '" target="_blank" rel="noopener">open raw</a>';
    try {
      const r = await fetch(BUCKET + f);
      if (!r.ok) throw new Error("HTTP " + r.status);
      const text = await r.text();
      if (f.endsWith(".txt")) body.innerHTML = '<div class="md-tools">' + raw + '</div><pre class="raw">' + M.esc(text) + "</pre>";
      else if (M.parseTraceTurns(text).turns.length) body.innerHTML = '<div class="trace-box">' + M.renderTrace(text, raw) + "</div>";
      else body.innerHTML = '<div class="md-tools">' + raw + '</div><div class="md">' + M.md(text) + "</div>";
    } catch (err) {
      body.innerHTML = '<p class="callout warn">Could not read ' + M.esc(f) + ": " + M.esc(err.message || err) + "</p>";
    }
  }

  document.addEventListener("click", (ev) => M.handleTraceClick(ev));
  show();
})();
