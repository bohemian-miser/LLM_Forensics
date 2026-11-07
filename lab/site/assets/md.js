// Safe Markdown and turn-by-turn trace rendering, shared by held-out.html and view.html.
// Browser: window.LLMFMarkdown; node: module.exports.
(function () {
  "use strict";
  const CLAMP_LINES = 12;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
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

  // rawLink: HTML for a link to the unrendered text, shown above the trace.
  function renderTrace(rawTrace, rawLink) {
    const { pre, turns } = parseTraceTurns(rawTrace);
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

  function setTurnExpanded(turnEl, expand) {
    if (expand) turnEl.classList.remove("is-clamped");
    else if (turnEl.querySelector(".turn-more")) turnEl.classList.add("is-clamped");
    turnEl.querySelectorAll(".turn-body details").forEach((d) => { d.open = expand; });
    const headBtn = turnEl.querySelector(".turn-head [data-turn-toggle]");
    if (headBtn) headBtn.textContent = expand ? "collapse" : (headBtn.dataset.expandLabel || "expand");
  }

  // Expand/collapse buttons in rendered traces; returns true when the click was one of them.
  function handleTraceClick(ev) {
    const turnBtn = ev.target && ev.target.closest && ev.target.closest("button[data-turn-toggle]");
    if (turnBtn) {
      const turnEl = turnBtn.closest(".turn");
      if (!turnEl) return true;
      const clamped = turnEl.classList.contains("is-clamped");
      const anyClosedFold = [...turnEl.querySelectorAll(".turn-body details")].some((d) => !d.open);
      setTurnExpanded(turnEl, clamped || anyClosedFold);
      return true;
    }
    const foldBtn = ev.target && ev.target.closest && ev.target.closest("button[data-fold]");
    if (foldBtn) {
      const box = foldBtn.closest(".trace-box");
      if (!box) return true;
      const expand = foldBtn.dataset.fold === "open";
      box.querySelectorAll(".turn").forEach((t) => setTurnExpanded(t, expand));
      box.querySelectorAll(".md details").forEach((d) => { d.open = expand; });
      return true;
    }
    return false;
  }

  const api = { esc, safeHref, inlineMd, md, parseTraceTurns, renderTrace, setTurnExpanded, handleTraceClick };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else window.LLMFMarkdown = api;
})();
