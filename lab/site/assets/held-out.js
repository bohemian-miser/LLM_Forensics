// held-out.html: the private held-out edits, decrypted in the browser. The bucket holds only AES-256-GCM ciphertext
// (private/heldout.enc); the key comes in the link's #k= fragment, which browsers never send to a server, and is kept
// in this browser's localStorage. Without the key the page shows nothing. Refreshes every 5 minutes.
(function () {
  "use strict";
  const URL_ENC = "https://storage.googleapis.com/llmf-edits-spectacle-game/private/heldout.enc";
  const STORE = "llmf-heldout-key";
  const CLAMP_LINES = 12;
  let lastSealed = "";
  let rawTexts = [];
  let liveBlobUrls = [];
  let didInitialJump = false;

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

  function registerRaw(text) {
    const idx = rawTexts.length;
    rawTexts.push(String(text == null ? "" : text));
    return idx;
  }

  function pageLinkWithKey(key, sid) {
    if (typeof location === "undefined") {
      return "held-out.html" + (sid ? "?s=" + encodeURIComponent(sid) : "") + (key ? "#k=" + encodeURIComponent(key) : "");
    }
    const u = new URL(location.href);
    u.hash = "";
    if (sid) u.searchParams.set("s", sid);
    else u.searchParams.delete("s");
    if (key) u.hash = "k=" + key;
    return u.toString();
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

  // ------------------------------------------------------------------ turn-by-turn trace renderer

  function prettifyToolCallMd(body) {
    // Convert single-line JSON Bash/Write tool calls into readable multi-line blocks while keeping metadata.
    const m = body.match(/^(\*\*→ tool call `([^`]+)`\*\*)\n(`{3,})json\n([\s\S]*?)\n\3\s*$/);
    if (!m) return { body, badge: null };
    const tool = m[2];
    let parsed = null;
    try { parsed = JSON.parse(m[4]); } catch (e) { parsed = null; }
    if (!parsed || typeof parsed !== "object") return { body, badge: "→ " + tool };
    if (tool === "Bash" && typeof parsed.command === "string") {
      const extras = Object.keys(parsed).filter((k) => k !== "command").map((k) => k + "=" + JSON.stringify(parsed[k]));
      const ticks = "`".repeat(Math.max(3, ...((parsed.command.match(/`+/g) || []).map((x) => x.length + 1))));
      const meta = extras.length ? " _(" + extras.join(", ") + ")_" : "";
      return {
        badge: "→ Bash",
        body: "**→ `Bash`**" + meta + "\n" + ticks + "bash\n" + parsed.command + "\n" + ticks,
      };
    }
    if (tool === "Write" && typeof parsed.file_path === "string" && typeof parsed.content === "string") {
      const ticks = "`".repeat(Math.max(3, ...((parsed.content.match(/`+/g) || []).map((x) => x.length + 1))));
      return {
        badge: "→ Write " + parsed.file_path.split("/").pop(),
        body: "**→ `Write`** `" + parsed.file_path + "`\n" + ticks + "\n" + parsed.content + "\n" + ticks,
      };
    }
    if (tool === "Read" && typeof parsed.file_path === "string") {
      const extras = Object.keys(parsed).filter((k) => k !== "file_path").map((k) => k + "=" + JSON.stringify(parsed[k]));
      const meta = extras.length ? " _(" + extras.join(", ") + ")_" : "";
      return {
        badge: "→ Read " + parsed.file_path.split("/").pop(),
        body: "**→ `Read`** `" + parsed.file_path + "`" + meta,
      };
    }
    return { body, badge: "→ " + tool };
  }

  function parseTraceTurns(raw) {
    const lines = String(raw == null ? "" : raw).replace(/\r\n?/g, "\n").split("\n");
    const pre = [], turns = [];
    let cur = null, fence = null;

    for (let i = 0; i < lines.length; i++) {
      const ln = lines[i];
      const stepMatch = ln.match(/^### (\d+)\.\s+(Claude|tool output)\s*$/) ||
                        ln.match(/^## (\d+)\.\s+(assistant|user|tool_results|system|tool)\s*$/);
      if (stepMatch) {
        fence = null;
        if (cur) turns.push(cur);
        cur = { num: stepMatch[1], who: stepMatch[2], kind: stepMatch[2] === "Claude" || stepMatch[2] === "assistant" ? "claude" : "tool", lines: [] };
        continue;
      }
      if (!fence) {
        const fm = ln.match(/^ {0,3}(`{3,}|~{3,})/);
        if (fm) fence = fm[1];
      } else {
        const closeRe = new RegExp("^ {0,3}" + fence[0] + "{" + fence.length + ",}\\s*$");
        if (closeRe.test(ln)) fence = null;
      }
      const resMatch = !fence && ln.match(/^## (Session result|Final report)\s*$/);
      if (resMatch) {
        if (cur) turns.push(cur);
        cur = { num: null, who: resMatch[1], kind: "result", lines: [] };
      } else if (cur) {
        cur.lines.push(ln);
      } else {
        pre.push(ln);
      }
    }
    if (cur) turns.push(cur);
    return { pre: pre.join("\n").trim(), turns };
  }

  function renderTurnBox(t) {
    let rawBody = t.lines.join("\n").trim();
    let badge = "";
    if (t.kind === "tool") {
      rawBody = rawBody.replace(/^\*\*← tool result\*\*\s*\n?/, "");
    } else if (t.kind === "claude") {
      // Split multi-block Claude turns (e.g. <details>thinking</details> + **→ tool call `Bash`**) so prettifyToolCallMd works.
      const tcIdx = rawBody.indexOf("**→ tool call `");
      if (tcIdx > 0) {
        const headPart = rawBody.slice(0, tcIdx).trim();
        const pretty = prettifyToolCallMd(rawBody.slice(tcIdx).trim());
        rawBody = headPart + "\n\n" + pretty.body;
        badge = pretty.badge || "";
      } else {
        const pretty = prettifyToolCallMd(rawBody);
        rawBody = pretty.body;
        if (pretty.badge) badge = pretty.badge;
      }
      if (!badge) {
        const tm = rawBody.match(/\*\*→ tool call `([^`]+)`\*\*/);
        if (tm) badge = "→ " + tm[1];
        else if (/<summary>thinking<\/summary>/i.test(rawBody)) badge = "thinking";
      }
    } else if (t.kind === "result") {
      const firstLine = t.lines.find((l) => l.trim()) || "";
      if (/^turns \d+/.test(firstLine.trim())) {
        badge = firstLine.trim();
        const idx = t.lines.indexOf(firstLine);
        rawBody = t.lines.slice(idx + 1).join("\n").trim();
      }
    }

    const lineCount = rawBody ? rawBody.split("\n").length : 0;
    const htmlBody = md(rawBody);
    const hasInnerFold = htmlBody.includes("<details");
    const clamp = lineCount > CLAMP_LINES;
    const whoLabel = t.kind === "tool" ? "← tool output" : esc(t.who);
    const numHtml = t.num != null ? '<span class="turn-num">#' + esc(t.num) + "</span>" : "";
    const badgeHtml = badge ? ' <span class="turn-badge">' + esc(badge) + "</span>" : "";
    const metaLines = lineCount > 1 ? '<span class="turn-lines">' + lineCount + " lines</span>" : "";
    const toggleBtn = (clamp || hasInnerFold)
      ? '<button type="button" class="link-btn turn-toggle" data-turn-toggle data-expand-label="expand (' + lineCount + ' lines)">' +
        (clamp ? "expand (" + lineCount + " lines)" : "expand") + "</button>"
      : "";

    return '<div class="turn turn-' + t.kind + (clamp ? " is-clamped" : "") + '">' +
      '<div class="turn-head"><div>' + numHtml + '<span class="turn-who">' + whoLabel + "</span>" + badgeHtml + "</div>" +
      '<div class="turn-actions">' + metaLines + (metaLines && toggleBtn ? " · " : "") + toggleBtn + "</div></div>" +
      '<div class="turn-body md">' + htmlBody + "</div>" +
      (clamp ? '<div class="turn-more"><button type="button" class="link-btn" data-turn-toggle data-expand-label="expand (' + lineCount + ' lines)">show all ' + lineCount + " lines</button></div>" : "") +
      "</div>";
  }

  function renderTrace(rawTrace, rawId) {
    const { pre, turns } = parseTraceTurns(rawTrace);
    const rawLink = '<a href="#" data-raw-id="' + rawId + '" target="_blank" rel="noopener">open raw in new tab</a>';
    if (!turns.length) {
      return '<div class="md-tools">' + rawLink + '</div><div class="md">' + md(rawTrace) + "</div>";
    }
    const preLines = pre ? pre.split("\n").filter((l) => l.trim()) : [];
    const title = preLines[0] ? preLines[0].replace(/^#+\s*/, "") : "";
    const sub = preLines.slice(1).join(" · ");
    const metaHtml = title
      ? '<div class="trace-meta"><strong>' + inlineMd(title) + "</strong>" +
        (sub ? '<div class="sub">' + inlineMd(sub.length > 180 ? sub.slice(0, 180) + "…" : sub) + "</div>" : "") + "</div>"
      : "";
    const tools = '<div class="md-tools"><span>' + turns.length + " turns</span> · " +
      '<button type="button" class="link-btn" data-fold="open">expand all</button> · ' +
      '<button type="button" class="link-btn" data-fold="close">collapse all</button> · ' +
      rawLink + "</div>";
    return tools + metaHtml + '<div class="turns">' + turns.map(renderTurnBox).join("") + "</div>";
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
    let h = '<details class="sid-detail"><summary>scan history and analyst report</summary>';
    if (scans.length) {
      h += '<div class="scroll"><table><thead><tr><th scope="col">Commit</th><th scope="col">Layer</th><th scope="col">Subject</th>' +
        '<th scope="col">Target</th><th scope="col">Top findings</th></tr></thead><tbody>' + scans.map((s) =>
          "<tr><td><code>" + esc(s.commit) + "</code></td><td>" + yn(s.layer_hit) + "</td><td>" + yn(s.subject_hit) +
          "</td><td>" + yn(s.target_hit) + "</td><td>" + (s.top || []).map((t) => esc(t.severity) + " " + esc(t.detector) + " L" + esc(t.layer)).join("<br>") +
          "</td></tr>").join("") + "</tbody></table></div>";
    }
    an.forEach((a) => {
      if (a.report) {
        const repRawId = registerRaw(a.report);
        h += "<details><summary>analyst report (" + esc(a.run) + ')</summary><div class="md-tools">' +
          '<a href="#" data-raw-id="' + repRawId + '" target="_blank" rel="noopener">open raw in new tab</a></div>' +
          '<div class="md">' + md(a.report) + "</div></details>";
      }
      if (a.trace) {
        const trRawId = registerRaw(a.trace);
        h += '<details class="trace-box"><summary>analyst trace</summary>' + renderTrace(a.trace, trRawId) + "</details>";
      }
    });
    return h + "</details>";
  }

  function render(d, key) {
    rawTexts = [];
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
        const rowId = "s-" + sid.replace(/[^A-Za-z0-9_-]/g, "-");
        const sidHref = pageLinkWithKey(key, sid);
        const det = detail(sid, d);
        return '<tr id="' + esc(rowId) + '"' + (win ? ' class="hl"' : "") + '><th scope="row"><code>' + esc(sid) + "</code>" +
          (win ? '<div><span class="st st-yes">analyst caught it</span></div>' : "") +
          '<div class="sub">' + esc(e.base.split("/").pop()) + ' · <a href="' + esc(sidHref) + '" data-copy-link title="Copy link to ' + esc(sid) + ' with key">copy link</a></div></th><td>' +
          esc(e.method.toUpperCase()) + " · layer" + (e.layers.length > 1 ? "s " : " ") +
          esc(e.layers.join(", ")) + "</td><td><code>" + esc(e.subject) + "</code> → <code>" + esc(e.target) + '</code><div class="sub">' + esc(e.prompt) +
          "</div></td><td>" + claimHtml(sc[sid]) + "</td><td>" + claimHtml(an[sid], queue[sid]) + "</td></tr>" +
          (det ? '<tr data-detail-for="' + esc(rowId) + '"><td></td><td colspan="4">' + det + "</td></tr>" : "");
      }).join("");
      return '<section class="card"><h2>' + esc(set) + '</h2><p class="hint">' + esc(summary) + '</p><div class="scroll"><table><thead><tr>' +
        ["Suspect", "Method", "Edit", "Scanner (latest commit)", "Claude Code analyst"].map((h) => '<th scope="col">' + h + "</th>").join("") +
        "</tr></thead><tbody>" + rows + "</tbody></table></div></section>";
    }).join("");
  }

  function hydrateRawLinks(root) {
    if (typeof URL === "undefined" || typeof URL.createObjectURL !== "function" || typeof Blob === "undefined") return;
    liveBlobUrls.forEach((u) => URL.revokeObjectURL(u));
    liveBlobUrls = [];
    root.querySelectorAll("a[data-raw-id]").forEach((a) => {
      const txt = rawTexts[Number(a.dataset.rawId)];
      if (txt == null) return;
      const u = URL.createObjectURL(new Blob([txt], { type: "text/plain;charset=utf-8" }));
      liveBlobUrls.push(u);
      a.href = u;
    });
  }

  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }

  function setTurnExpanded(turnEl, expand) {
    if (expand) turnEl.classList.remove("is-clamped");
    else if (turnEl.querySelector(".turn-more")) turnEl.classList.add("is-clamped");
    turnEl.querySelectorAll(".turn-body details").forEach((d) => { d.open = expand; });
    const headBtn = turnEl.querySelector(".turn-head [data-turn-toggle]");
    if (headBtn) headBtn.textContent = expand ? "collapse" : (headBtn.dataset.expandLabel || "expand");
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
      const fullKeyHref = pageLinkWithKey(key, null);
      status.className = "hint";
      status.innerHTML = "Sealed " + esc(blob.sealed) + " · " + Object.keys(d.edits).length + " held-out edits · refreshes every 5 minutes · " +
        '<a href="' + esc(fullKeyHref) + '" data-copy-link>copy link with key</a> · ' +
        '<a href="#" id="ho-forget">forget the key in this browser</a>';
      document.getElementById("ho-forget").onclick = (ev) => { ev.preventDefault(); store("del"); location.reload(); };
      const openIdx = [...body.querySelectorAll("details")].map((el) => el.open);
      body.innerHTML = render(d, key);
      hydrateRawLinks(body);
      body.querySelectorAll("details").forEach((el, idx) => { if (openIdx[idx]) el.open = true; });
      if (!didInitialJump) {
        didInitialJump = true;
        const targetSid = new URLSearchParams(location.search).get("s");
        if (targetSid) {
          const rowId = "s-" + targetSid.replace(/[^A-Za-z0-9_-]/g, "-");
          const row = document.getElementById(rowId);
          const detRow = body.querySelector('tr[data-detail-for="' + CSS.escape(rowId) + '"] details.sid-detail');
          if (detRow) detRow.open = true;
          if (row) row.scrollIntoView({ block: "start" });
        }
      }
    } catch (err) {
      status.className = "callout warn";
      status.textContent = (err && err.name === "OperationError") ? "The saved key doesn't open this data. Open the page from the current link." :
        "Could not read the held-out data: " + (err && err.message || err);
    }
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { md, inlineMd, parseTraceTurns, renderTrace, render, detail, analystClaim, scannerClaim, pageLinkWithKey };
  } else if (typeof document !== "undefined") {
    document.addEventListener("click", (ev) => {
      const copyEl = ev.target && ev.target.closest && ev.target.closest("a[data-copy-link]");
      if (copyEl) {
        ev.preventDefault();
        const orig = copyEl.textContent;
        copyText(copyEl.href).then(() => {
          copyEl.textContent = "copied!";
          setTimeout(() => { copyEl.textContent = orig; }, 1500);
        }).catch(() => {});
        return;
      }
      const turnBtn = ev.target && ev.target.closest && ev.target.closest("button[data-turn-toggle]");
      if (turnBtn) {
        const turnEl = turnBtn.closest(".turn");
        if (!turnEl) return;
        const clamped = turnEl.classList.contains("is-clamped");
        const anyClosedFold = [...turnEl.querySelectorAll(".turn-body details")].some((d) => !d.open);
        setTurnExpanded(turnEl, clamped || anyClosedFold);
        return;
      }
      const foldBtn = ev.target && ev.target.closest && ev.target.closest("button[data-fold]");
      if (foldBtn) {
        const box = foldBtn.closest("details.trace-box");
        if (!box) return;
        const expand = foldBtn.dataset.fold === "open";
        box.querySelectorAll(".turn").forEach((t) => setTurnExpanded(t, expand));
        box.querySelectorAll(".md details").forEach((d) => { d.open = expand; });
      }
    });
    takeKeyFromHash();
    refresh();
    setInterval(refresh, 300000);
  }
})();
