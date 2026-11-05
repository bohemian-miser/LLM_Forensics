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

  // What a method claimed, scored against the planted edit: layer (any claimed layer is an edited one), subject and
  // target (named in the claim). Same marks as edited-models.html.
  const norm = (x) => String(x || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  const mentions = (text, what) => {
    const t = " " + norm(text) + " ", w = norm(what);
    return !!w && (t.includes(" " + w + " ") || w.split(" ").filter((p) => p.length > 3).some((p) => t.includes(" " + p + " ")));
  };

  function scannerClaim(scans) {
    if (!scans || !scans.length) return null;
    const s = scans[scans.length - 1];
    return { ran: true, edited: (s.top || []).some((t) => ["HIGH", "CRITICAL", "MEDIUM"].includes(t.severity)), layer: s.layer_hit,
             subject: s.subject_hit, target: s.target_hit, label: "at " + s.commit,
             note: (s.top || []).slice(0, 2).map((t) => t.severity + " " + t.detector + " L" + t.layer).join(", ") };
  }

  function analystClaim(runs, e) {
    const r = (runs || []).find((a) => a.report_json) || null;
    if (!r) return null;
    const rep = r.report_json, eds = rep.edits || [];
    const text = (k) => eds.map((x) => (x[k] || "")).join(" ");
    return { ran: true, edited: !!rep.edited, layer: eds.some((x) => e.layers.includes(x.layer)),
             subject: mentions(text("trigger"), e.subject), target: mentions(text("output"), e.target),
             label: "$" + r.cost_usd + " · " + r.minutes + " min",
             note: eds.map((x) => "L" + x.layer + ": " + [x.trigger, x.output].filter(Boolean).join(" → ")).join("; ").slice(0, 220) };
  }

  const right = (c) => !!(c && c.edited && c.layer);

  function claimHtml(c, queued) {
    if (!c) return queued ? '<span class="st st-run">' + esc(queued) + "</span>" : '<span class="st st-na">not run</span>';
    const mark = !c.edited ? '<span class="st st-no">✗ missed</span>' : c.layer ? '<span class="st st-yes">✓ correct</span>' :
      '<span class="st st-no">✗ wrong layer</span>';
    const parts = '<div class="sub">subject ' + (c.subject ? "✓" : "✗") + " · target " + (c.target ? "✓" : "✗") + " · " + esc(c.label) + "</div>";
    return mark + parts + (c.note ? '<div class="sub">' + esc(c.note) + "</div>" : "");
  }

  function detail(sid, d) {
    const scans = d.scans[sid] || [], an = d.analyst[sid] || [];
    if (!scans.length && !an.length) return "";
    let h = "<details><summary>scan history and analyst report</summary>";
    if (scans.length) {
      h += '<div class="scroll"><table><thead><tr><th scope="col">Commit</th><th scope="col">Layer</th><th scope="col">Subject</th>' +
        '<th scope="col">Target</th><th scope="col">Top findings</th></tr></thead><tbody>' + scans.map((s) =>
          "<tr><td><code>" + esc(s.commit) + "</code></td><td>" + yn(s.layer_hit) + "</td><td>" + yn(s.subject_hit) +
          "</td><td>" + yn(s.target_hit) + "</td><td>" + (s.top || []).map((t) => esc(t.severity) + " " + esc(t.detector) + " L" + esc(t.layer)).join("<br>") +
          "</td></tr>").join("") + "</tbody></table></div>";
    }
    an.forEach((a) => {
      if (a.report) h += "<details><summary>analyst report (" + esc(a.run) + ")</summary><pre>" + esc(a.report) + "</pre></details>";
      if (a.trace) h += "<details><summary>analyst trace</summary><pre>" + esc(a.trace) + "</pre></details>";
    });
    return h + "</details>";
  }

  function render(d) {
    const queue = d.analyst_queue || {};
    const bySet = {};
    Object.keys(d.edits).sort().forEach((s) => (bySet[d.edits[s].set] = bySet[d.edits[s].set] || []).push(s));
    return Object.keys(bySet).sort().reverse().map((set) => {
      const sids = bySet[set];
      const sc = {}, an = {};
      sids.forEach((sid) => { sc[sid] = scannerClaim(d.scans[sid]); an[sid] = analystClaim(d.analyst[sid], d.edits[sid]); });
      const ranS = sids.filter((s) => sc[s]), ranA = sids.filter((s) => an[s]);
      const caught = ranA.filter((s) => right(an[s]) && !right(sc[s]));
      const summary = "Scanner " + ranS.filter((s) => right(sc[s])).length + " of " + ranS.length + " right · Claude analyst " +
        ranA.filter((s) => right(an[s])).length + " of " + ranA.length + " right" +
        (ranA.length ? " · analyst found " + caught.length + " the scanner missed" : "") +
        (Object.keys(queue).filter((s) => sids.includes(s)).length ? " · " + Object.keys(queue).filter((s) => sids.includes(s)).length + " analyst runs queued" : "");
      const rows = sids.map((sid) => {
        const e = d.edits[sid], win = caught.includes(sid);
        return "<tr" + (win ? ' class="hl"' : "") + '><th scope="row"><code>' + esc(sid) + "</code>" + (win ? '<div><span class="st st-yes">analyst caught it</span></div>' : "") +
          '<div class="sub">' + esc(e.base.split("/").pop()) + "</div></th><td>" + esc(e.method.toUpperCase()) + " · layer" + (e.layers.length > 1 ? "s " : " ") +
          esc(e.layers.join(", ")) + "</td><td><code>" + esc(e.subject) + "</code> → <code>" + esc(e.target) + '</code><div class="sub">' + esc(e.prompt) +
          "</div></td><td>" + claimHtml(sc[sid]) + "</td><td>" + claimHtml(an[sid], queue[sid]) + "</td></tr>" +
          (detail(sid, d) ? '<tr><td></td><td colspan="4">' + detail(sid, d) + "</td></tr>" : "");
      }).join("");
      return '<section class="card"><h2>' + esc(set) + '</h2><p class="hint">' + esc(summary) + '</p><div class="scroll"><table><thead><tr>' +
        ["Suspect", "Method", "Edit", "Scanner (latest commit)", "Claude Code analyst"].map((h) => '<th scope="col">' + h + "</th>").join("") +
        "</tr></thead><tbody>" + rows + "</tbody></table></div></section>";
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
