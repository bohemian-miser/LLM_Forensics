// Edited models: the browser lists the public bucket and reads each suspect's small JSON files,
// so a running batch shows up here with nothing to commit. Objects are fetched by the generation
// the listing reports, which keys the browser cache per version: a changed file is refetched,
// an unchanged one is not. Under node, `module.exports` exposes the loader for tests.
(function () {
  const BUCKET = "llmf-edits-spectacle-game", PREFIX = "edits";
  const API = "https://storage.googleapis.com/storage/v1/b/" + BUCKET + "/o";
  const PUBLIC = "https://storage.googleapis.com/" + BUCKET + "/";
  const REFRESH_MS = 60000;
  const TIERS = [["detectors", "Coded detectors"], ["analyst", "Claude Code analyst"],
                 ["investigators", "LLM investigators"]];
  const SEVERITY = { INFO: 0, LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4 };
  const DIR_ORDER = ["weights", "config", "agents", "samples"];
  // agents/ subdirectories small enough to list file by file (LLM investigator transcripts, module runs)
  const EXPAND_DIRS = ["investigators", "modules", "scan-with-base"];
  // agents/ files linked from the table's tier cells
  const TIER_FILES = {
    analyst: [["report", "agents/report.md"], ["trace", "agents/trace.md"], ["scratch", "agents/scratch/index.md"]],
    detectors: [["scan", "agents/scan.txt"]],
    investigators: [["gemini", "agents/investigators/llm-gemini.md"], ["gemini-raw", "agents/investigators/llm-gemini-raw.md"],
                    ["openai", "agents/investigators/llm-openai.md"], ["openai-raw", "agents/investigators/llm-openai-raw.md"],
                    ["anthropic", "agents/investigators/llm-anthropic.md"], ["anthropic-raw", "agents/investigators/llm-anthropic-raw.md"]],
  };
  const AGENT_FILES = ["report.md", "trace.md", "scan.txt", "prompt.md", "report.json", "scan.json", "trace.jsonl"];
  const parsed = new Map(); // "name@generation" -> parsed body, kept across refreshes

  // ------------------------------------------------------------------ bucket access

  async function getJson(fetchFn, url) {
    const r = await fetchFn(url);
    if (!r.ok) throw new Error("HTTP " + r.status + " listing " + url);
    return r.json();
  }

  async function list(fetchFn, prefix, delimiter) {
    const items = [], prefixes = [];
    let token;
    do {
      let url = API + "?prefix=" + encodeURIComponent(prefix) +
        "&fields=items(name,size,updated,generation),prefixes,nextPageToken";
      if (delimiter) url += "&delimiter=" + encodeURIComponent(delimiter);
      if (token) url += "&pageToken=" + encodeURIComponent(token);
      const page = await getJson(fetchFn, url);
      items.push(...(page.items || []));
      prefixes.push(...(page.prefixes || []));
      token = page.nextPageToken;
    } while (token);
    return { items, prefixes };
  }

  function objectUrl(name, generation) {
    const url = PUBLIC + name.split("/").map(encodeURIComponent).join("/");
    return generation ? url + "?generation=" + generation : url;
  }

  function listingUrl(prefix) {
    return API + "?prefix=" + encodeURIComponent(prefix) + "&fields=items(name,size,updated)";
  }

  async function object(fetchFn, item, asText) {
    const key = item.name + "@" + item.generation;
    if (!parsed.has(key)) {
      const r = await fetchFn(objectUrl(item.name, item.generation));
      if (!r.ok) throw new Error("HTTP " + r.status + " for " + item.name);
      parsed.set(key, asText ? await r.text() : await r.json());
    }
    return parsed.get(key);
  }

  // ------------------------------------------------------------------ one suspect

  async function loadSuspect(fetchFn, sid) {
    const dir = PREFIX + "/" + sid + "/";
    const { items } = await list(fetchFn, dir);
    const files = {}; // path inside the suspect -> listing entry
    for (const it of items) files[it.name.slice(dir.length)] = it;
    const get = (path, asText) => (files[path] ? object(fetchFn, files[path], asText) : null);
    const [control, manifest, ev, report, scan] = await Promise.all([
      get("config/control.json"), get("config/edit_manifest.json"), get("config/edit_eval.json"),
      get("agents/report.json"), get("agents/scan.json")]);
    const exit = report ? null : await get("agents/exit.txt", true);

    const sus = { sid, dir, files, control: !!control, model: null, method: null, layers: [], edits: [],
                  efficacy: null, evalSummary: null, report, scan, exit: null,
                  updated: items.reduce((m, it) => (it.updated > m ? it.updated : m), "") };
    if (control) {
      sus.model = control.model || null;
    } else if (manifest) {
      sus.model = manifest.model || null;
      sus.method = manifest.method || null;
      sus.layers = (manifest.layers || []).filter(Number.isInteger);
      const reqs = (manifest.spec && manifest.spec.requests) ||
        (manifest.edits || []).map((e) => e.request).filter(Boolean);
      sus.edits = reqs.map((r) => ({ prompt: r.prompt, subject: r.subject, target_new: r.target_new,
                                     target_true: r.target_true }));
    }
    if (ev) {
      const first = (ev.efficacy || [])[0], sum = ev.summary || {};
      const prob = (side) => (((first || {})[side] || {}).target_new || {}).prob;
      const before = prob("before"), after = prob("after");
      if (typeof after === "number") sus.efficacy = { before, after };
      else if (typeof sum.efficacy_prob_after === "number") {
        sus.efficacy = { before: sum.efficacy_prob_before, after: sum.efficacy_prob_after };
      }
      sus.evalSummary = sum;
    }
    if (typeof exit === "string") {
      const m = /exit=(\d+)/.exec(exit);
      if (m) sus.exit = Number(m[1]);
    }
    sus.tiers = { detectors: scanTier(scan, false), analyst: analystTier(report), investigators: scanTier(scan, true) };
    return sus;
  }

  function statusOf(sus) {
    const paths = Object.keys(sus.files);
    const has = (p) => paths.some((x) => x.startsWith(p));
    let s;
    if (!has("agents/")) s = "queued";
    else if (sus.report) s = "analysed";
    else if (sus.exit === 124) s = "analyst timed out, no report";
    else if (sus.exit !== null) s = "analyst exited " + sus.exit + ", no report";
    else if (has("agents/live/")) s = "analyst running";
    else s = "analyst started";
    return sus.scan ? s + " · scanned" : s;
  }

  // ------------------------------------------------------------------ tiers and scoring

  const unique = (xs) => xs.filter((x, i) => xs.indexOf(x) === i);
  const isLlm = (id) => /^llm-/.test(id || "");
  const clip = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  // A tier's claim: {edited, layers, note}; null when it did not run.
  function analystTier(report) {
    if (!report) return null;
    const edits = report.edits || [];
    let note = "";
    if (report.edited) note = edits.map((e) => [e.trigger, e.output].filter(Boolean).join(" → ")).join("; ");
    else if (typeof report.confidence_unmodified === "number") {
      note = "P(unmodified) " + report.confidence_unmodified.toFixed(2);
    }
    return { edited: !!report.edited, layers: unique(edits.map((e) => e.layer).filter(Number.isInteger)),
             note: clip(note, 110) };
  }

  // llm=false: the weight analysers; llm=true: the llm-* investigators. MEDIUM and above count.
  function scanTier(scan, llm) {
    if (!scan) return null;
    const ran = scan.detectors || [];
    if (ran.length && !ran.some((d) => isLlm(d.id) === llm)) return null;
    const hits = (scan.findings || []).filter(
      (f) => (SEVERITY[f.severity] || 0) >= SEVERITY.MEDIUM && isLlm(f.detector) === llm);
    const note = unique(hits.map((f) => f.detector + " " + f.severity)).concat(
      ran.filter((d) => isLlm(d.id) === llm && d.status && d.status !== "OK")
         .map((d) => d.id + " " + String(d.status).toLowerCase()));
    return { edited: hits.length > 0, layers: unique(hits.map((f) => f.layer).filter(Number.isInteger)),
             note: note.join(", ") };
  }

  function verdict(tier) {
    if (!tier.edited) return "unedited";
    return "edited: layer " + (tier.layers.length ? tier.layers.join(", ") : "?");
  }

  function truthOf(sus) {
    if (sus.control) return { edited: false, layers: [] };
    if (sus.method) return { edited: true, layers: sus.layers };
    return null;
  }

  // {correct: true|false|null, mark}; null when the tier did not run or the truth is unknown.
  function score(tier, truth) {
    if (!tier || !truth) return { correct: null, mark: "" };
    if (tier.edited !== truth.edited) return { correct: false, mark: truth.edited ? "missed" : "false alarm" };
    if (truth.edited && !tier.layers.some((l) => truth.layers.includes(l))) return { correct: false, mark: "wrong layer" };
    return { correct: true, mark: "correct" };
  }

  function scored(sus) {
    const truth = truthOf(sus), out = {};
    for (const [k] of TIERS) out[k] = Object.assign({ tier: sus.tiers[k] }, score(sus.tiers[k], truth));
    return out;
  }

  function summarize(suspects) {
    const out = {};
    for (const [k] of TIERS) out[k] = { right: 0, scored: 0 };
    for (const sus of suspects) {
      if (sus.error) continue;
      const sc = scored(sus);
      for (const [k] of TIERS) {
        if (sc[k].correct === null) continue;
        out[k].scored += 1;
        if (sc[k].correct) out[k].right += 1;
      }
    }
    return out;
  }

  async function load(fetchFn) {
    const top = await list(fetchFn, PREFIX + "/", "/");
    const sids = top.prefixes.map((p) => p.slice(PREFIX.length + 1, -1)).filter(Boolean).sort();
    const suspects = await Promise.all(
      sids.map((sid) => loadSuspect(fetchFn, sid).catch((err) => ({ sid, error: String(err && err.message || err) }))));
    return { suspects, summary: summarize(suspects) };
  }

  // ------------------------------------------------------------------ rendering

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function tok(s) { // a token with its leading space made visible
    if (s == null) return "–";
    s = String(s);
    return '<code class="tok">' + esc(s.startsWith(" ") ? "·" + s.slice(1) : s) + "</code>";
  }

  function pct(v) {
    if (typeof v !== "number") return "–";
    return (v > 0 && v < 0.01 ? (v * 100).toFixed(1) : Math.round(v * 100)) + "%";
  }

  function size(bytes) {
    const n = Number(bytes);
    if (!(n >= 0)) return "";
    if (n < 1e3) return n + " B";
    if (n < 1e6) return (n / 1e3).toFixed(0) + " kB";
    if (n < 1e9) return (n / 1e6).toFixed(0) + " MB";
    return (n / 1e9).toFixed(1) + " GB";
  }

  function hhmm(d) { return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }

  function when(iso) {
    if (!iso) return "";
    const d = new Date(iso), now = new Date();
    return d.toDateString() === now.toDateString() ? hhmm(d)
      : d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + hhmm(d);
  }

  function editHtml(ed) {
    const subj = ed.subject || "?";
    const prompt = esc(ed.prompt || "{}").replace("{}", "<mark>" + esc(subj) + "</mark>");
    const old = ed.target_true ? "<s>" + tok(ed.target_true) + "</s> → " : "";
    return prompt + " " + old + tok(ed.target_new);
  }

  function efficacyHtml(sus) {
    if (!sus.efficacy) return '<span class="hint">–</span>';
    let out = "P(new) " + pct(sus.efficacy.before) + " → " + pct(sus.efficacy.after);
    const sum = sus.evalSummary || {};
    if (typeof sum.efficacy_argmax_rate === "number") {
      out += '<div class="sub">top-1 ' + pct(sum.efficacy_argmax_rate);
      if (typeof sum.paraphrase_argmax_rate === "number") out += " · paraphrases " + pct(sum.paraphrase_argmax_rate);
      out += "</div>";
    }
    return out;
  }

  function methodHtml(sus) {
    if (sus.control) return "none (control)";
    if (!sus.method) return '<span class="hint">?</span>';
    const word = sus.layers.length > 1 ? "layers" : "layer";
    return esc(sus.method.toUpperCase()) + (sus.layers.length ? " · " + word + " " + sus.layers.join(", ") : "");
  }

  function tierHtml(sc) {
    if (!sc.tier) return '<span class="st st-na">not run</span>';
    let mark = "";
    if (sc.correct === true) mark = '<span class="st st-yes">✓ correct</span> ';
    else if (sc.correct === false) mark = '<span class="st st-no">✗ ' + esc(sc.mark) + "</span> ";
    return mark + esc(verdict(sc.tier)) + (sc.tier.note ? '<div class="sub">' + esc(sc.tier.note) + "</div>" : "");
  }

  function statusHtml(sus) {
    if (sus.error) return '<span class="st st-no">load failed</span><div class="sub">' + esc(sus.error) + "</div>";
    const s = statusOf(sus);
    const cls = /running/.test(s) ? "st-run" : /^analysed/.test(s) ? "st-dim" : /^queued/.test(s) ? "st-na" : "st-part";
    return '<span class="st ' + cls + '">' + esc(s) + "</span>" +
      (sus.updated ? '<div class="sub">changed ' + esc(when(sus.updated)) + "</div>" : "");
  }

  function filesHtml(sus) {
    const groups = {}; // top-level dir -> {files: [{name, it}], subdirs: {name: count}}
    for (const [path, it] of Object.entries(sus.files)) {
      const i = path.indexOf("/");
      if (i < 0) continue;
      const dir = path.slice(0, i), rest = path.slice(i + 1);
      const g = groups[dir] || (groups[dir] = { files: [], subdirs: {} });
      const j = rest.indexOf("/");
      if (j < 0) g.files.push({ name: rest, it });
      else g.subdirs[rest.slice(0, j)] = (g.subdirs[rest.slice(0, j)] || 0) + 1;
    }
    const dirs = DIR_ORDER.filter((d) => groups[d]).concat(Object.keys(groups).filter((d) => !DIR_ORDER.includes(d)).sort());
    const rank = (name) => { const i = AGENT_FILES.indexOf(name); return i < 0 ? AGENT_FILES.length : i; };
    const rows = dirs.map((dir) => {
      const g = groups[dir];
      g.files.sort((a, b) => rank(a.name) - rank(b.name) || a.name.localeCompare(b.name));
      const links = g.files.map(({ name, it }) => {
        const big = Number(it.size) >= 1e6 ? ' <span class="hint">' + size(it.size) + "</span>" : "";
        return '<a href="' + esc(objectUrl(it.name)) + '" title="' + esc(size(it.size)) + '">' + esc(name) + "</a>" + big;
      });
      for (const sub of Object.keys(g.subdirs).sort()) {
        const inner = Object.entries(sus.files).filter(([p]) => p.startsWith(dir + "/" + sub + "/") && p.split("/").length === 3);
        if (EXPAND_DIRS.includes(sub) && inner.length) {
          for (const [p, it] of inner.sort()) {
            links.push('<a href="' + esc(objectUrl(it.name)) + '">' + esc(p.slice(dir.length + 1)) + "</a>");
          }
        } else if (sus.files[dir + "/" + sub + "/index.md"]) {
          links.push('<a href="' + esc(objectUrl(sus.files[dir + "/" + sub + "/index.md"].name)) + '">' + esc(sub) +
            '/index.md</a> <span class="hint">' + g.subdirs[sub] + " files</span>");
        } else {
          links.push('<a href="' + esc(listingUrl(sus.dir + dir + "/" + sub + "/")) + '">' + esc(sub) +
            '/</a> <span class="hint">' + g.subdirs[sub] + " files</span>");
        }
      }
      return "<dt>" + esc(dir) + "/</dt><dd>" + links.join(" · ") + "</dd>";
    });
    return rows.length ? '<dl class="facts files">' + rows.join("") + "</dl>" : '<p class="hint">No files yet.</p>';
  }

  function notesHtml(sus) {
    const rep = sus.report;
    if (!rep) return "";
    const edits = (rep.edits || []).map((ed) => {
      const head = [ed.layer != null ? "layer " + ed.layer : "", ed.tensor ? "<code>" + esc(ed.tensor) + "</code>" : "",
                    ed.kind ? esc(ed.kind) : ""].filter(Boolean).join(" · ");
      const evidence = ed.evidence == null ? [] : Array.isArray(ed.evidence) ? ed.evidence : [ed.evidence];
      const ev = evidence.map((x) => "<li>" + esc(typeof x === "object" ? JSON.stringify(x) : x) + "</li>").join("");
      return '<div class="edit"><div class="edit-h">' + head + "</div>" +
        (ed.trigger || ed.output ? "<p>" + esc(ed.trigger || "?") + " → " + esc(ed.output || "?") + "</p>" : "") +
        (ev ? "<ul>" + ev + "</ul>" : "") + "</div>";
    }).join("");
    const rejected = (rep.rejected_candidates || []).map((c) =>
      "<li>layer " + esc(c.layer) + ": " + esc(c.verdict || "") + (c.flagged_by ? ' <span class="hint">(flagged by ' + esc(c.flagged_by) + ")</span>" : "") + "</li>").join("");
    return "<details><summary>Analyst notes</summary>" +
      (rep.notes ? '<p class="notes">' + esc(rep.notes) + "</p>" : "") + edits +
      (rejected ? "<h5>Rejected candidates</h5><ul>" + rejected + "</ul>" : "") + "</details>";
  }

  // Quote bucket/model metadata as shell arguments, including prompts with apostrophes.
  const shellQuote = (s) => "'" + String(s).replace(/'/g, "'\\''") + "'";

  function recreate(sus) {
    const hasWeights = Object.keys(sus.files).some((p) => p.startsWith("weights/"));
    if (!sus.model || (!sus.control && !hasWeights)) return "# Model weights are not available yet.";
    const source = sus.control ? sus.model : "gs://" + BUCKET + "/" + sus.dir + "weights";
    const ed = sus.edits[0];
    const prompt = ed && ed.prompt && ed.subject ? ed.prompt.replace("{}", ed.subject) : "Hello!";
    return ["# Interactive chat (downloads are cached)",
      "llmf chat " + shellQuote(source),
      "", "# Try your own prompt",
      "llmf chat " + shellQuote(source) + " --prompt " + shellQuote(prompt) + (ed ? " --raw" : ""),
      "", "# Scan with local detectors",
      "llmf scan " + shellQuote(source) + " --no-llm",
      "", "# Optional: compare against the base model too",
      "llmf scan " + shellQuote(source) + " --base " + shellQuote(sus.model) + " --no-llm"
    ].join("\n");
  }

  function cardHtml(sus) {
    const id = "s-" + sus.sid.replace(/[^A-Za-z0-9_-]/g, "-");
    if (sus.error) {
      return '<section class="card" id="' + id + '"><h2><code>' + esc(sus.sid) + "</code></h2>" +
        '<p class="callout warn">Could not load this suspect: ' + esc(sus.error) + "</p></section>";
    }
    const sc = scored(sus);
    const tiers = TIERS.map(([k, label]) => "<dt>" + esc(label) + "</dt><dd>" + tierHtml(sc[k]) + "</dd>").join("");
    return '<section class="card" id="' + id + '">' +
      "<h2><code>" + esc(sus.sid) + "</code> " + statusHtml(sus).replace(/<div class="sub">.*$/, "") + "</h2>" +
      '<dl class="facts">' +
      "<dt>model</dt><dd><code>" + esc(sus.model || "?") + "</code></dd>" +
      "<dt>method</dt><dd>" + methodHtml(sus) + "</dd>" +
      "<dt>edit</dt><dd>" + (sus.edits.map(editHtml).join("<br>") || "none") + "</dd>" +
      "<dt>efficacy</dt><dd>" + (sus.edits.length ? efficacyHtml(sus) : "–") + "</dd>" +
      tiers + "</dl>" + notesHtml(sus) +
      "<h4>Files</h4>" + filesHtml(sus) +
      "<h4>Use it</h4><p class=\"hint\">Assumes a working <code>llmf</code> install and <code>python</code> from the same environment,. " +
      '<a href="https://github.com/bohemian-miser/LLM_Forensics#install">GitHub · installation instructions</a>.</p>' +
      "<p class=\"hint\">Loads the published weights directly; delta metadata identifies the base model. " +
      "Downloads are cached on disk, and registered local base models are reused. No standalone model save is required. " +
      "Scans below use local detectors; omit <code>--no-llm</code> to include investigators with configured API keys.</p><pre>" + esc(recreate(sus)) + "</pre></section>";
  }

  function rowHtml(sus) {
    const id = "s-" + sus.sid.replace(/[^A-Za-z0-9_-]/g, "-");
    const head = '<th scope="row"><a href="#' + id + '"><code>' + esc(sus.sid) + "</code></a>" +
      '<div class="sub">' + esc(sus.model || "?") + "</div></th>";
    if (sus.error) return "<tr>" + head + "<td>" + statusHtml(sus) + '</td><td colspan="6"></td></tr>';
    const sc = scored(sus);
    return "<tr>" + head + "<td>" + statusHtml(sus) + "</td><td>" + methodHtml(sus) + "</td>" +
      "<td>" + (sus.edits.map(editHtml).join("<br>") || "none") + "</td>" +
      '<td class="eff">' + (sus.edits.length ? efficacyHtml(sus) : "–") + "</td>" +
      TIERS.map(([k]) => "<td>" + tierHtml(sc[k]) + tierFilesHtml(sus, k) + "</td>").join("") + "</tr>";
  }

  function tierFilesHtml(sus, tier) {
    const links = (TIER_FILES[tier] || []).filter(([, p]) => sus.files[p])
      .map(([label, p]) => '<a href="' + esc(objectUrl(sus.files[p].name)) + '">' + esc(label) + "</a>");
    return links.length ? '<div class="sub files">' + links.join(" · ") + "</div>" : "";
  }

  function summaryText(summary) {
    return TIERS.map(([k, label]) => label + " " + summary[k].right + " of " + summary[k].scored + " right").join(" · ");
  }

  function render(model) {
    const head = "<thead><tr>" + ["Suspect", "Status", "Method", "Edit", "Efficacy"].concat(TIERS.map((t) => t[1]))
      .map((h) => '<th scope="col">' + esc(h) + "</th>").join("") + "</tr></thead>";
    const body = model.suspects.map(rowHtml).join("") ||
      '<tr><td colspan="8" class="c-na">No suspects in the bucket yet.</td></tr>';
    setHtml(document.getElementById("em-table"), "<table>" + head + "<tbody>" + body + "</tbody></table>");
    renderCards(document.getElementById("em-cards"), model.suspects);
    const summary = model.suspects.length ? summaryText(model.summary) : "";
    const sumEl = document.getElementById("em-summary");
    if (sumEl.textContent !== summary) sumEl.textContent = summary;
  }

  // A refresh rewrites only what changed, so open notes, selections and the scroll position survive it.
  const lastHtml = new WeakMap();
  function setHtml(el, html) {
    if (lastHtml.get(el) === html) return;
    el.innerHTML = html;
    lastHtml.set(el, html);
  }

  function renderCards(box, suspects) {
    const keep = new Set();
    let prev = null;
    for (const sus of suspects) {
      const html = cardHtml(sus), id = "s-" + sus.sid.replace(/[^A-Za-z0-9_-]/g, "-");
      let el = document.getElementById(id);
      if (!el || lastHtml.get(el) !== html) {
        const open = el ? [...el.querySelectorAll("details")].map((d) => d.open) : [];
        const tmp = document.createElement("div");
        tmp.innerHTML = html;
        const fresh = tmp.firstElementChild;
        fresh.querySelectorAll("details").forEach((d, i) => { if (open[i]) d.open = true; });
        if (el) el.replaceWith(fresh);
        el = fresh;
        lastHtml.set(el, html);
      }
      if (el.parentNode !== box || el.previousElementSibling !== prev) box.insertBefore(el, prev ? prev.nextSibling : box.firstChild);
      keep.add(el);
      prev = el;
    }
    [...box.children].forEach((c) => { if (!keep.has(c)) c.remove(); });
  }

  function main() {
    const status = document.getElementById("em-status");
    let busy = false;
    async function refresh() {
      if (busy) return;
      busy = true;
      try {
        const model = await load((u) => fetch(u));
        render(model);
        status.className = "hint";
        status.textContent = model.suspects.length + " suspect" + (model.suspects.length === 1 ? "" : "s") +
          " · updated " + hhmm(new Date()) + " · refreshes every minute";
      } catch (err) {
        status.className = "callout warn";
        status.textContent = "Could not update edited models from gs://" + BUCKET + ": " + (err && err.message || err) + ". Retrying every minute.";
      } finally {
        busy = false;
      }
    }
    refresh();
    setInterval(refresh, REFRESH_MS);
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { load, notesHtml, statusOf, scored, summarize, summaryText, objectUrl, BUCKET, PREFIX };
  } else if (typeof document !== "undefined") {
    main();
  }
})();
