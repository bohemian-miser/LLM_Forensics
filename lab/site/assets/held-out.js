// held-out.html: the private held-out edits, decrypted in the browser. The bucket holds only AES-256-GCM ciphertext
// (private/heldout.enc); the key comes in the link's #k= fragment, which browsers never send to a server, and is kept
// in this browser's localStorage. Without the key the page shows nothing. Refreshes every 5 minutes.
(function () {
  "use strict";
  const URL_ENC = "https://storage.googleapis.com/llmf-edits-spectacle-game/private/heldout.enc";
  const STORE = "llmf-heldout-key";
  let lastSealed = "";

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function yn(b) {
    return b == null ? '<span class="st st-na">–</span>' : b ? '<span class="st st-yes">yes</span>' : '<span class="st st-no">no</span>';
  }
  function b64(s) {
    s = s.replace(/-/g, "+").replace(/_/g, "/");
    while (s.length % 4) s += "=";
    return Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
  }
  function store(op, v) {
    try { return op === "get" ? localStorage.getItem(STORE) : op === "set" ? localStorage.setItem(STORE, v) : localStorage.removeItem(STORE); }
    catch (e) { return null; }
  }

  function takeKeyFromHash() {
    const m = location.hash.match(/k=([A-Za-z0-9_-]+)/);
    if (!m) return;
    store("set", m[1]);
    history.replaceState(null, "", location.pathname + location.search);  // keep the key out of the address bar
  }

  async function decrypt(blob, key) {
    const k = await crypto.subtle.importKey("raw", b64(key), "AES-GCM", false, ["decrypt"]);
    const pt = await crypto.subtle.decrypt({ name: "AES-GCM", iv: b64(blob.iv), additionalData: new TextEncoder().encode("llmf-heldout-v1") }, k, b64(blob.ct));
    return JSON.parse(new TextDecoder().decode(pt));
  }

  function scanCells(list) {
    if (!list || !list.length) return '<td colspan="3"><span class="hint">not scanned</span></td>';
    const s = list[list.length - 1];
    return "<td>" + yn(s.layer_hit) + "</td><td>" + yn(s.subject_hit) + "</td><td>" + yn(s.target_hit) + "</td>";
  }

  function detail(sid, d) {
    const e = d.edits[sid], scans = d.scans[sid] || [], an = d.analyst[sid] || [];
    let h = '<details><summary>' + scans.length + " scan(s) · " + an.length + " analyst run(s)</summary>";
    if (scans.length) {
      h += '<div class="scroll"><table><thead><tr><th scope="col">Commit</th><th scope="col">When</th><th scope="col">Layer</th><th scope="col">Subject</th>' +
        '<th scope="col">Target</th><th scope="col">Top findings</th></tr></thead><tbody>' + scans.map((s) =>
          "<tr><td><code>" + esc(s.commit) + "</code></td><td>" + esc(s.when) + "</td><td>" + yn(s.layer_hit) + "</td><td>" + yn(s.subject_hit) +
          "</td><td>" + yn(s.target_hit) + "</td><td>" + (s.top || []).map((t) => esc(t.severity) + " " + esc(t.detector) + " L" + esc(t.layer)).join("<br>") +
          "</td></tr>").join("") + "</tbody></table></div>";
    }
    an.forEach((a) => {
      h += '<h4>Analyst <code>' + esc(a.run) + "</code> · $" + esc(a.cost_usd) + " · " + esc(a.minutes) + " min</h4>";
      if (a.report) h += '<details><summary>report.md</summary><pre>' + esc(a.report) + "</pre></details>";
      if (a.trace) h += '<details><summary>trace</summary><pre>' + esc(a.trace) + "</pre></details>";
    });
    return h + "</details>";
  }

  function render(d) {
    const sids = Object.keys(d.edits).sort((a, b) => (d.edits[a].set + a).localeCompare(d.edits[b].set + b));
    const bySet = {};
    sids.forEach((s) => (bySet[d.edits[s].set] = bySet[d.edits[s].set] || []).push(s));
    return Object.keys(bySet).sort().reverse().map((set) => {
      const rows = bySet[set].map((sid) => {
        const e = d.edits[sid], an = (d.analyst[sid] || [])[0];
        const verdict = an && an.report_json && an.report_json.edits && an.report_json.edits[0];
        return '<tr><th scope="row"><code>' + esc(sid) + "</code></th><td>" + esc(e.method.toUpperCase()) + " L" + esc(e.layers.join(",")) +
          '<div class="sub">' + esc(e.base.split("/").pop()) + "</div></td><td><code>" + esc(e.subject) + "</code> → <code>" + esc(e.target) +
          '</code><div class="sub">' + esc(e.prompt) + "</div></td>" + scanCells(d.scans[sid]) + "<td>" +
          (verdict ? "L" + esc(verdict.layer) + '<div class="sub">' + esc(verdict.trigger || "").slice(0, 140) + "</div>" : '<span class="hint">–</span>') +
          "</td></tr><tr><td></td><td colspan=\"6\">" + detail(sid, d) + "</td></tr>";
      }).join("");
      return '<section class="card"><h2>' + esc(set) + '</h2><div class="scroll"><table><thead><tr><th scope="col">Suspect</th><th scope="col">Planted</th>' +
        '<th scope="col">Subject → target</th><th scope="col">Layer found</th><th scope="col">Subject found</th><th scope="col">Target found</th>' +
        '<th scope="col">Analyst</th></tr></thead><tbody>' + rows + "</tbody></table></div></section>";
    }).join("");
  }

  async function refresh() {
    const status = document.getElementById("ho-status"), body = document.getElementById("ho-body");
    const key = store("get");
    if (!key) {
      status.className = "callout"; status.textContent = "This page needs its key. Open it from the link with #k=… once; this browser then remembers it.";
      return;
    }
    try {
      const r = await fetch(URL_ENC + "?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const blob = await r.json();
      if (blob.sealed === lastSealed) return;
      const d = await decrypt(blob, key);
      lastSealed = blob.sealed;
      status.className = "hint";
      status.innerHTML = "Sealed " + esc(blob.sealed) + " · " + Object.keys(d.edits).length + " held-out edits · refreshes every 5 minutes · " +
        '<a href="#" id="ho-forget">forget the key in this browser</a>';
      document.getElementById("ho-forget").onclick = (ev) => { ev.preventDefault(); store("del"); location.reload(); };
      body.innerHTML = render(d);
    } catch (err) {
      status.className = "callout warn";
      status.textContent = (err && err.name === "OperationError") ? "The saved key doesn't open this data. Open the page from the current link." :
        "Could not read the held-out data: " + (err && err.message || err);
    }
  }

  takeKeyFromHash();
  refresh();
  setInterval(refresh, 300000);
})();
