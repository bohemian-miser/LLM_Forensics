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

  // ------------------------------------------------------------------ safe Markdown + <details> renderer

  function safeHref(url) {
    const u = String(url || "").trim();
    return /^(https?:\/\/|mailto:|#|\/|[A-Za-z0-9_.-]+\.html(?:[?#].*)?$)/.test(u) ? u : null;
  }

  function inlineMd(raw) {
    const codes = [], escapes = [];
    // 1. Extract inline code spans (`...` or ``...``) verbatim before HTML-escaping or emphasis.
    let s = String(raw == null ? "" : raw).replace(/(`+)([\s\S]*?)\1/g, (_, ticks, body) => {
      const inner = body.length >= 2 && body.startsWith(" ") && body.endsWith(" ") && body.trim() ? body.slice(1, -1) : body;
      codes.push("<code>" + esc(inner) + "</code>");
      return "\u0000C" + (codes.length - 1) + "\u0000";
    });
    // 2. Protect backslash-escaped ASCII punctuation (\*, \_, \|, \[, etc.).
    s = s.replace(/\\([\\`*_~[\]()#+-.!|<>])/g, (_, ch) => {
      escapes.push(esc(ch));
      return "\u0000E" + (escapes.length - 1) + "\u0000";
    });
    // 3. Escape all remaining HTML so arbitrary tokens (e.g. '<small', '<script>') never inject DOM nodes.
    s = esc(s);
    // 4. Links [label](url) with scheme allowlist.
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (full, label, href) => {
      const clean = safeHref(href.replace(/&amp;/g, "&"));
      return clean ? '<a href="' + esc(clean) + '">' + label + "</a>" : full;
    });
    // 5. Bold, italic, strikethrough.
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/~~(.+?)~~/g, "<s>$1</s>");
    s = s.replace(/(^|[^\w*])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
    // 6. Restore backslash escapes and inline code spans.
    s = s.replace(/\u0000E(\d+)\u0000/g, (_, i) => escapes[Number(i)]);
    s = s.replace(/\u0000C(\d+)\u0000/g, (_, i) => codes[Number(i)]);
    return s;
  }

  function splitTableRow(line) {
    let s = line.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    const cells = [];
    let cur = "", ticks = 0;
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (c === "\\" && i + 1 < s.length) { cur += c + s[++i]; continue; }
      if (c === "`") { ticks = ticks ? 0 : 1; cur += c; continue; }
      if (c === "|" && !ticks) { cells.push(cur.trim()); cur = ""; continue; }
      cur += c;
    }
    cells.push(cur.trim());
    return cells;
  }

  function isTableSep(line) {
    if (!line || !line.includes("|")) return null;
    const cells = splitTableRow(line);
    if (!cells.length || !cells.every((c) => /^:?-{3,}:?$/.test(c))) return null;
    return cells.map((c) => (c.startsWith(":") && c.endsWith(":") ? "center" : c.endsWith(":") ? "right" : ""));
  }

  function findDetailsClose(lines, startIdx) {
    let depth = 0, fence = null;
    for (let i = startIdx; i < lines.length; i++) {
      const ln = lines[i];
      if (fence) {
        if (fence.re.test(ln)) fence = null;
        continue;
      }
      const fm = ln.match(/^ {0,3}(`{3,}|~{3,})/);
      if (fm) {
        fence = { re: new RegExp("^ {0,3}" + fm[1][0] + "{" + fm[1].length + ",}[ \\t]*$") };
        continue;
      }
      const opens = (ln.match(/<details\b[^>]*>/gi) || []).length;
      const closes = (ln.match(/<\/details\s*>/gi) || []).length;
      depth += opens - closes;
      if (depth <= 0) return i;
    }
    return -1;
  }

  function renderList(lines, startIdx) {
    const itemRe = /^(\s*)([-*+]|\d+\.)\s+(.*)$/;
    const first = lines[startIdx].match(itemRe);
    const baseIndent = first[1].length;
    const ordered = /\d+\./.test(first[2]);
    const startNum = ordered ? parseInt(first[2], 10) : 1;
    const tag = ordered ? "ol" : "ul";
    const attr = ordered && startNum !== 1 ? ' start="' + startNum + '"' : "";
    const items = [];
    let i = startIdx;

    while (i < lines.length) {
      const m = lines[i].match(itemRe);
      if (!m || m[1].length < baseIndent) break;
      if (m[1].length > baseIndent) {
        const sub = renderList(lines, i);
        if (items.length) items[items.length - 1].extra.push(sub.html);
        else items.push({ head: "", extra: [sub.html] });
        i = sub.next;
        continue;
      }
      if ((/\d+\./.test(m[2])) !== ordered) break;
      const headLines = [m[3]];
      const extra = [];
      i++;
      while (i < lines.length) {
        const ln = lines[i];
        if (!ln.trim()) {
          if (i + 1 < lines.length && (lines[i + 1].match(/^(\s+)/) || ["", ""])[1].length > baseIndent) { i++; continue; }
          break;
        }
        const nm = ln.match(itemRe);
        if (nm) {
          if (nm[1].length <= baseIndent) break;
          const sub = renderList(lines, i);
          extra.push(sub.html);
          i = sub.next;
          continue;
        }
        const ind = (ln.match(/^(\s*)/) || ["", ""])[1].length;
        if (ind > baseIndent) { headLines.push(ln.trim()); i++; }
        else break;
      }
      items.push({ head: headLines.join(" "), extra });
    }
    return {
      html: "<" + tag + attr + ">" + items.map((it) => "<li>" + inlineMd(it.head) + it.extra.join("") + "</li>").join("") + "</" + tag + ">",
      next: i,
    };
  }

  function md(src) {
    const lines = String(src == null ? "" : src).replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;

    while (i < lines.length) {
      const ln = lines[i];
      if (!ln.trim()) { i++; continue; }

      // 1. Fenced code block (``` or ~~~, CommonMark closing length >= opening length).
      const fm = ln.match(/^ {0,3}(`{3,}|~{3,})[ \t]*([^\s`~]*)/);
      if (fm) {
        const closeRe = new RegExp("^ {0,3}" + fm[1][0] + "{" + fm[1].length + ",}[ \\t]*$");
        const lang = fm[2] ? ' class="lang-' + esc(fm[2]) + '"' : "";
        const buf = [];
        i++;
        while (i < lines.length && !closeRe.test(lines[i])) buf.push(lines[i++]);
        if (i < lines.length) i++;
        out.push("<pre><code" + lang + ">" + esc(buf.join("\n")) + "</code></pre>");
        continue;
      }

      // 2. Safe <details><summary>...</summary>...</details> block (for thinking & truncated tool outputs).
      const dm = ln.match(/^\s*<details(?:\s+open)?>\s*(?:<summary>([\s\S]*?)<\/summary>\s*(.*))?$/i);
      if (dm) {
        const endIdx = findDetailsClose(lines, i);
        if (endIdx === i && dm[1] != null && /<\/details\s*>\s*$/i.test(dm[2] || "")) {
          const inner = dm[2].replace(/<\/details\s*>\s*$/i, "");
          out.push('<details class="md-fold"><summary>' + inlineMd(dm[1]) + '</summary><div class="md-fold-body">' + md(inner) + "</div></details>");
          i++;
          continue;
        }
        if (endIdx > i) {
          let summary = dm[1], bodyStart = i + 1, firstRest = dm[2] || "";
          if (summary == null && i + 1 <= endIdx) {
            const sm = lines[i + 1].match(/^\s*<summary>([\s\S]*?)<\/summary>\s*(.*)$/i);
            if (sm) { summary = sm[1]; firstRest = sm[2] || ""; bodyStart = i + 2; }
          }
          const innerLines = (firstRest ? [firstRest] : []).concat(lines.slice(bodyStart, endIdx));
          const last = lines[endIdx].replace(/<\/details\s*>\s*$/i, "");
          if (last.trim()) innerLines.push(last);
          out.push('<details class="md-fold"><summary>' + inlineMd(summary != null ? summary : "details") +
            '</summary><div class="md-fold-body">' + md(innerLines.join("\n")) + "</div></details>");
          i = endIdx + 1;
          continue;
        }
      }

      // 3. ATX Heading (# .. ######).
      const hm = ln.match(/^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
      if (hm) {
        const lvl = hm[1].length;
        out.push("<h" + lvl + ">" + inlineMd(hm[2]) + "</h" + lvl + ">");
        i++;
        continue;
      }

      // 4. Horizontal rule.
      if (/^ {0,3}(?:-\s*){3,}$|^ {0,3}(?:\*\s*){3,}$|^ {0,3}(?:_\s*){3,}$/.test(ln)) {
        out.push("<hr>");
        i++;
        continue;
      }

      // 5. GFM Pipe table.
      const aligns = i + 1 < lines.length && ln.includes("|") ? isTableSep(lines[i + 1]) : null;
      if (aligns) {
        const head = splitTableRow(ln);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim() && lines[i].includes("|") && !/^ {0,3}(`{3,}|~{3,}|#{1,6}\s)/.test(lines[i])) {
          rows.push(splitTableRow(lines[i++]));
        }
        const styleAt = (k) => (aligns[k] ? ' style="text-align:' + aligns[k] + '"' : "");
        const ths = head.map((c, k) => '<th scope="col"' + styleAt(k) + ">" + inlineMd(c) + "</th>").join("");
        const trs = rows.map((r) => "<tr>" + head.map((_, k) => "<td" + styleAt(k) + ">" + inlineMd(r[k] != null ? r[k] : "") + "</td>").join("") + "</tr>").join("");
        out.push('<div class="scroll"><table><thead><tr>' + ths + "</tr></thead><tbody>" + trs + "</tbody></table></div>");
        continue;
      }

      // 6. Blockquote (> ...).
      if (/^ {0,3}>\s?/.test(ln)) {
        const bq = [];
        while (i < lines.length && /^ {0,3}>\s?/.test(lines[i])) bq.push(lines[i++].replace(/^ {0,3}>\s?/, ""));
        out.push("<blockquote>" + md(bq.join("\n")) + "</blockquote>");
        continue;
      }

      // 7. Unordered / ordered list (with nested sublists).
      if (/^\s*([-*+]|\d+\.)\s+/.test(ln)) {
        const lst = renderList(lines, i);
        out.push(lst.html);
        i = lst.next;
        continue;
      }

      // 8. Paragraph.
      const para = [];
      while (i < lines.length) {
        const cur = lines[i];
        if (!cur.trim()) break;
        if (para.length && (
          /^ {0,3}(`{3,}|~{3,})/.test(cur) ||
          /^\s*<details\b/i.test(cur) ||
          /^ {0,3}#{1,6}\s+/.test(cur) ||
          /^ {0,3}(?:-\s*){3,}$|^ {0,3}(?:\*\s*){3,}$|^ {0,3}(?:_\s*){3,}$/.test(cur) ||
          /^ {0,3}>\s?/.test(cur) ||
          /^\s*([-*+]|\d+\.)\s+/.test(cur) ||
          (i + 1 < lines.length && cur.includes("|") && isTableSep(lines[i + 1]))
        )) break;
        para.push(cur);
        i++;
      }
      const rendered = para.map((p, idx) => {
        const br = idx < para.length - 1 && /  $/.test(p);
        return inlineMd(p.trim()) + (br ? "<br>" : "");
      }).join("\n");
      out.push("<p>" + rendered + "</p>");
    }
    return out.join("\n");
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
      if (a.report) {
        h += "<details><summary>analyst report (" + esc(a.run) + ')</summary><div class="md">' + md(a.report) + "</div></details>";
      }
      if (a.trace) {
        const rendered = md(a.trace);
        const tools = rendered.includes("<details")
          ? '<div class="md-tools"><button type="button" class="link-btn" data-fold="open">expand all</button> · ' +
            '<button type="button" class="link-btn" data-fold="close">collapse all</button></div>'
          : "";
        h += '<details class="trace-box"><summary>analyst trace</summary>' + tools + '<div class="md">' + rendered + "</div></details>";
      }
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
      const openIdx = [...body.querySelectorAll("details")].map((el) => el.open);
      body.innerHTML = render(d);
      body.querySelectorAll("details").forEach((el, idx) => { if (openIdx[idx]) el.open = true; });
    } catch (err) {
      status.className = "callout warn";
      status.textContent = (err && err.name === "OperationError") ? "The saved key doesn't open this data. Open the page from the current link." :
        "Could not read the held-out data: " + (err && err.message || err);
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { md, inlineMd, render, detail, analystClaim, scannerClaim };
  } else if (typeof document !== "undefined") {
    document.addEventListener("click", (ev) => {
      const btn = ev.target && ev.target.closest && ev.target.closest("button[data-fold]");
      if (!btn) return;
      const box = btn.closest("details.trace-box");
      if (!box) return;
      const open = btn.dataset.fold === "open";
      box.querySelectorAll(".md details").forEach((d) => { d.open = open; });
    });
    takeKeyFromHash();
    refresh();
    setInterval(refresh, 300000);
  }
})();
