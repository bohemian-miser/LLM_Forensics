#!/usr/bin/env python3
"""Build the static results site from results/*/ into site/_build/ (stdlib only).

    python site/build.py                         # -> site/_build/index.html
    python site/build.py --include-examples      # also show results/_example
    python -m http.server -d site/_build 8000    # preview

Reads results/<name>/summary.json (written by `llmf evaluate` / `llmf all`) plus the
per-method report JSONs next to it. If summary.json is missing it is reconstructed
without ground-truth scores. Directories starting with "_" are example fixtures: they
are shown only with --include-examples or when there are no real experiments.

No third-party imports: GitHub Actions builds this with a bare Python.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_pipeline():
    """Load llm_forensics/pipeline.py by path so the package __init__ (which may grow heavy
    imports) is never executed; pipeline.py itself is stdlib-only."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_llmf_pipeline", ROOT / "llm_forensics" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

REPO_URL = "https://github.com/bohemian-miser/LLM_Forensics"
METHOD_LABELS = {
    "diff": ("Weight diff", "needs original"),
    "investigator_with_original": ("Claude investigator", "needs original"),
    "blind": ("Blind statistics", "original-free"),
    "literature": ("Literature (Youssef et al.)", "original-free"),
    "investigator_blind": ("Claude investigator", "original-free"),
}
METHOD_ORDER = list(METHOD_LABELS)

e = html.escape


# --------------------------------------------------------------------------- data loading

def read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_experiments(results: Path, include_examples: bool) -> list[dict[str, Any]]:
    exps = []
    for d in sorted(p for p in results.iterdir() if p.is_dir()) if results.exists() else []:
        summary = read_json(d / "summary.json")
        if summary is None:
            if not any(d.glob("*.json")):
                continue
            summary = _load_pipeline().summarize_dir(d, None, None, {"name": d.name})
        summary.setdefault("name", d.name)
        summary["example"] = bool(summary.get("example")) or d.name.startswith("_")
        summary["_dir"] = d
        summary["_reports"] = {k: read_json(d / m["file"]) for k, m in summary.get("methods", {}).items()}
        exps.append(summary)
    real = [x for x in exps if not x["example"]]
    if real and not include_examples:
        return real
    return exps


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-_") or "experiment"


def gt_first(s: dict[str, Any]) -> dict[str, Any]:
    gts = s.get("ground_truth") or []
    return gts[0] if gts else {}


def gt_prompt(s: dict[str, Any]) -> tuple[str | None, str | None]:
    raw = s.get("ground_truth_raw") or {}
    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    items = raw.get("requests") or raw.get("edits") or [raw]
    it = items[0] if items else {}
    true = it.get("target_true")
    if isinstance(true, dict):
        true = true.get("str")
    return it.get("prompt"), true


# --------------------------------------------------------------------------- formatting

def fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 10 ** -nd or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.{nd}f}"
    return str(v)


def tok(s: Any) -> str:
    """Show a token with visible leading space."""
    if s is None:
        return "–"
    return f'<code class="tok">{e(str(s)).replace(" ", "&middot;", 1) if str(s).startswith(" ") else e(str(s))}</code>'


def hit(v: Any, label: str = "") -> str:
    """Status cell: icon + word, never colour alone."""
    if v is None:
        return f'<span class="st st-na" title="not applicable">– <span class="sr">{label} n/a</span></span>'
    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 1:
        return f'<span class="st st-part">◐ {v:.0%}</span>'
    ok = bool(v)
    return (f'<span class="st st-{"yes" if ok else "no"}">{"✓" if ok else "✗"} '
            f'<span class="st-word">{"hit" if ok else "miss"}</span></span>')


def method_label(key: str) -> str:
    name, cond = METHOD_LABELS.get(key, (key, ""))
    return f'{e(name)} <span class="cond cond-{"blind" if "free" in cond else "orig"}">{e(cond)}</span>'


# --------------------------------------------------------------------------- charts (inline SVG)

def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    t = math.floor(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def svg_bars_log(values: list[float], title: str, desc: str) -> str:
    """Singular values as columns on a log axis (rank-one edits show one tall bar)."""
    vals = [max(float(v), 1e-12) for v in values if v is not None]
    if not vals:
        return ""
    W, H, ml, mr, mt, mb = 560, 220, 56, 12, 14, 34
    pw, ph = W - ml - mr, H - mt - mb
    lo = math.floor(math.log10(min(vals)))
    hi = math.ceil(math.log10(max(vals)))
    hi = hi if hi > lo else lo + 1

    def y(v):
        return mt + ph - (math.log10(v) - lo) / (hi - lo) * ph

    band = pw / len(vals)
    bw = min(24, band * 0.6)
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{e(title)}"><title>{e(title)}</title>'
             f'<desc>{e(desc)}</desc>']
    for p in range(lo, hi + 1):
        yy = y(10 ** p)
        parts.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                     f'<text class="axis" x="{ml - 6}" y="{yy + 4:.1f}" text-anchor="end">1e{p}</text>')
    base = mt + ph
    for i, v in enumerate(vals):
        x = ml + band * i + (band - bw) / 2
        top = y(v)
        h = max(base - top, 1)
        r = min(4, h / 2, bw / 2)
        path = (f"M{x:.1f},{base:.1f} V{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} "
                f"H{x + bw - r:.1f} Q{x + bw:.1f},{top:.1f} {x + bw:.1f},{top + r:.1f} V{base:.1f} Z")
        cls = "bar hl" if i == 0 else "bar"
        parts.append(f'<g class="hov"><rect class="hit" x="{ml + band * i:.1f}" y="{mt}" width="{band:.1f}" '
                     f'height="{ph}"/><path class="{cls}" d="{path}"/><title>σ{i + 1} = {v:.4g}</title></g>')
        parts.append(f'<text class="axis" x="{x + bw / 2:.1f}" y="{base + 14}" text-anchor="middle">σ{i + 1}</text>')
    parts.append(f'<line class="baseline" x1="{ml}" x2="{W - mr}" y1="{base}" y2="{base}"/>')
    if len(vals) > 1:
        ratio = vals[0] / vals[1]
        parts.append(f'<text class="annot" x="{W - mr}" y="{mt + 10}" text-anchor="end">σ1/σ2 = {ratio:.3g}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_layer_line(rows: list[dict[str, Any]], key: str, title: str, gt_layers: list[int],
                   threshold: float | None = None) -> str:
    pts = [(int(r["layer"]), float(r[key])) for r in rows
           if isinstance(r.get("layer"), (int, float)) and isinstance(r.get(key), (int, float))]
    if len(pts) < 2:
        return ""
    W, H, ml, mr, mt, mb = 560, 220, 48, 14, 16, 34
    pw, ph = W - ml - mr, H - mt - mb
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts] + ([threshold] if threshold is not None else [])
    x0, x1 = min(xs), max(xs)
    ticks = _nice_ticks(min(ys), max(ys))
    y0, y1 = min(ticks[0], min(ys)), max(ticks[-1], max(ys))
    y1 = y1 if y1 > y0 else y0 + 1

    def X(v):
        return ml + (v - x0) / max(x1 - x0, 1) * pw

    def Y(v):
        return mt + ph - (v - y0) / (y1 - y0) * ph

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{e(title)}"><title>{e(title)}</title>']
    for t in ticks:
        parts.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                     f'<text class="axis" x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for gl in gt_layers:
        if x0 <= gl <= x1:
            parts.append(f'<line class="gtline" x1="{X(gl):.1f}" x2="{X(gl):.1f}" y1="{mt}" y2="{mt + ph}"/>'
                         f'<text class="annot gt" x="{X(gl) + 4:.1f}" y="{mt + 10}">edited layer {gl}</text>')
    if threshold is not None:
        parts.append(f'<line class="thresh" x1="{ml}" x2="{W - mr}" y1="{Y(threshold):.1f}" y2="{Y(threshold):.1f}"/>'
                     f'<text class="annot" x="{W - mr}" y="{Y(threshold) - 4:.1f}" text-anchor="end">threshold {threshold:g}</text>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(v):.1f}" for i, (x, v) in enumerate(pts))
    parts.append(f'<path class="line" d="{d}"/>')
    best = max(pts, key=lambda p: p[1])
    for x, v in pts:
        r = 4 if (x, v) == best or x in gt_layers else 2.5
        parts.append(f'<g class="hov"><circle class="hit" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="9"/>'
                     f'<circle class="dot{" hl" if (x, v) == best else ""}" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="{r}"/>'
                     f'<title>layer {x}: {key} = {v:.4g}</title></g>')
    step = max(1, (x1 - x0) // 8)
    for t in range(x0, x1 + 1, step):
        parts.append(f'<text class="axis" x="{X(t):.1f}" y="{mt + ph + 14}" text-anchor="middle">{t}</text>')
    parts.append(f'<text class="axis" x="{ml + pw / 2:.1f}" y="{H - 4}" text-anchor="middle">layer</text>')
    parts.append(f'<line class="baseline" x1="{ml}" x2="{W - mr}" y1="{mt + ph}" y2="{mt + ph}"/></svg>')
    return "".join(parts)


def pick_layer_key(rows: list[dict[str, Any]]) -> str | None:
    keys = [k for k in rows[0] if k != "layer" and isinstance(rows[0].get(k), (int, float))]
    for pref in ("robust_z", "_z", "score"):
        for k in keys:
            if k.endswith(pref):
                return k
    return keys[0] if keys else None


# --------------------------------------------------------------------------- page chrome

def page(title: str, body: str, depth: int = 0, description: str = "") -> str:
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(description or 'Planting and detecting ROME weight edits in open-weight LLMs')}">
<link rel="stylesheet" href="{up}assets/style.css">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ccircle cx='6.5' cy='6.5' r='4.5' fill='none' stroke='%232a78d6' stroke-width='2'/%3E%3Cpath d='M10 10l4.5 4.5' stroke='%232a78d6' stroke-width='2'/%3E%3C/svg%3E">
<script>try{{var t=localStorage.getItem("llmf-theme");if(t)document.documentElement.dataset.theme=t}}catch(_){{}}</script>
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href="{up}index.html">LLM&nbsp;Forensics</a>
  <nav><a href="{up}index.html#experiments">Experiments</a><a href="{up}index.html#methods">Methods</a><a href="{REPO_URL}">GitHub</a>
  <button id="theme" type="button" aria-label="Toggle light/dark theme" title="Toggle theme">◐</button></nav>
</div></header>
<main class="wrap">
{body}
</main>
<footer class="wrap foot">Built {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from committed <code>results/</code> ·
<a href="{REPO_URL}">source</a> · BSD-3-Clause</footer>
<script src="{up}assets/app.js"></script>
</body>
</html>
"""


# --------------------------------------------------------------------------- index

def score_cell(m: dict[str, Any] | None) -> str:
    if not m:
        return '<td class="c-na">not run</td>'
    ev = m.get("eval") or {}
    if not ev:
        return f'<td>{"detected" if m.get("detected") else "clean"} · {fmt(m.get("confidence"))}</td>'
    bits = [("L", ev.get("layer_correct"), "layer"), ("S", ev.get("subject_hit@10"), "subject in top 10"),
            ("T", ev.get("target_hit@10"), "target in top 10")]
    inner = "".join(
        f'<span class="pip pip-{"na" if v is None else ("yes" if v else "no")}" title="{t}: {fmt(v)}">{k}</span>'
        for k, v, t in bits)
    return f'<td><span class="pips">{inner}</span></td>'


def build_index(exps: list[dict[str, Any]]) -> str:
    any_example = any(x["example"] for x in exps)
    rows = []
    for x in exps:
        g = gt_first(x)
        badge = ' <span class="badge ex">example</span>' if x["example"] else ""
        cells = "".join(score_cell(x.get("methods", {}).get(k)) for k in METHOD_ORDER)
        ee = x.get("edit_eval") or {}
        eff = ee.get("efficacy_success", ee.get("efficacy_score"))
        rows.append(
            f'<tr><th scope="row"><a href="exp/{slug(x["name"])}.html">{e(x["name"])}</a>{badge}'
            f'<div class="sub">{e(str(x.get("base_model") or "?"))} · {e(g.get("subject") or "?")} → {tok(g.get("target"))}'
            f' · layer {", ".join(map(str, g.get("layers") or [])) or "?"}</div></th>'
            f'<td class="num">{f"{eff:.0%}" if isinstance(eff, (int, float)) else "–"}</td>{cells}</tr>')
    head = "".join(f"<th scope='col'>{method_label(k)}</th>" for k in METHOD_ORDER)

    # aggregate per method
    agg_rows = []
    for k in METHOD_ORDER:
        ms = [x["methods"][k] for x in exps if k in x.get("methods", {})]
        if not ms:
            continue

        def mean(key):
            vals = [(m.get("eval") or {}).get(key) for m in ms]
            vals = [float(v) for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        det = sum(bool(m.get("detected")) for m in ms) / len(ms)
        agg_rows.append(
            f"<tr><th scope='row'>{method_label(k)}</th><td class='num'>{len(ms)}</td><td class='num'>{det:.0%}</td>"
            + "".join(f"<td class='num'>{'–' if v is None else f'{v:.0%}'}</td>"
                      for v in (mean("layer_correct"), mean("subject_hit@1"), mean("subject_hit@10"),
                                mean("target_hit@1"), mean("target_hit@10"))) + "</tr>")

    example_note = ('<p class="callout warn"><strong>Example data.</strong> Experiments marked '
                    '<span class="badge ex">example</span> are synthetic fixtures used to develop this site, '
                    'not real measurements.</p>') if any_example else ""
    body = f"""
<section class="hero">
  <h1>Can you tell a model's memory was rewritten?</h1>
  <p class="lede">We plant targeted factual edits (ROME rank-one updates) into open-weight language models,
  then attack them with forensics: diffing against the original weights, original-free statistics from the
  literature, and a Claude agent that investigates the weights with tools. Every number on this site is scored
  against the editor's answer key.</p>
</section>
{example_note}
<section id="experiments">
  <h2>Experiments</h2>
  <p class="hint">Each cell shows whether the method recovered the edited
  <span class="pip pip-yes">L</span>ayer, the trigger <span class="pip pip-yes">S</span>ubject (top&nbsp;10) and the
  new <span class="pip pip-yes">T</span>arget (top&nbsp;10). Green ✓-style pips are hits, red are misses, grey is not
  applicable. Hover for details.</p>
  <div class="scroll"><table class="matrix">
  <thead><tr><th scope="col">Experiment</th><th scope="col">Edit<br>efficacy</th>{head}</tr></thead>
  <tbody>{''.join(rows) or '<tr><td colspan="7">No results yet — run <code>make all</code>.</td></tr>'}</tbody>
  </table></div>
</section>
<section id="methods">
  <h2>Methods at a glance</h2>
  <div class="scroll"><table>
  <thead><tr><th scope="col">Method</th><th scope="col">runs</th><th scope="col">flagged edit</th>
  <th scope="col">layer correct</th><th scope="col">subject@1</th><th scope="col">subject@10</th>
  <th scope="col">target@1</th><th scope="col">target@10</th></tr></thead>
  <tbody>{''.join(agg_rows)}</tbody></table></div>
  <dl class="defs">
    <dt>Weight diff</dt><dd>Subtract the original weights, take the SVD of each changed tensor, read the
    value direction through the unembedding (target) and match the key direction against candidate subjects.</dd>
    <dt>Blind statistics</dt><dd>No original: look for layers whose spectra are outliers relative to their neighbours.</dd>
    <dt>Literature</dt><dd>Published original-free detectors, e.g. Youssef et al. 2025 (arXiv:2505.20819).</dd>
    <dt>Claude investigator</dt><dd>An agent with tools for spectra, logit lens, unembedding and forward passes;
    run with and without access to the original model. Transcripts are on each experiment page.</dd>
  </dl>
</section>
"""
    return page("LLM Forensics", body)


# --------------------------------------------------------------------------- experiment page

def edit_eval_html(ee: Any) -> str:
    if not isinstance(ee, dict) or not ee:
        return '<p class="hint">No edit evaluation recorded.</p>'
    tiles, tables, gens = [], [], ""
    for k, v in ee.items():
        if k in ("example",):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            pct = 0 <= v <= 1 and ("success" in k or "score" in k or "acc" in k)
            tiles.append(f'<div class="tile"><div class="tv">{f"{v:.0%}" if pct else fmt(v)}</div>'
                         f'<div class="tl">{e(k.replace("_", " "))}</div></div>')
        elif isinstance(v, dict):
            flat = flatten(v)
            tables.append(f"<div><h4>{e(k.replace('_', ' '))}</h4><table class='kv'>" + "".join(
                f"<tr><th scope='row'>{e(kk)}</th><td class='num'>{fmt(vv, 4)}</td></tr>" for kk, vv in flat.items())
                + "</table></div>")
        elif isinstance(v, list) and v and isinstance(v[0], dict) and "prompt" in v[0]:
            cols = [c for c in v[0] if c != "prompt"]
            gens = ("<h4>Generations before / after</h4><div class='scroll'><table><thead><tr><th>prompt</th>"
                    + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr></thead><tbody>"
                    + "".join("<tr><td>" + e(str(g.get("prompt"))) + "</td>"
                              + "".join(f"<td class='gen'>{e(str(g.get(c, '')))}</td>" for c in cols) + "</tr>"
                              for g in v[:20]) + "</tbody></table></div>")
    return (f'<div class="tiles">{"".join(tiles)}</div>' if tiles else "") + \
        (f'<div class="kvgrid">{"".join(tables)}</div>' if tables else "") + gens


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif not isinstance(v, (list, tuple)):
            out[key] = v
    return out


def candidates(cands: list[dict], field: str, truth: str | None, matcher) -> str:
    if not cands:
        return '<span class="hint">none</span>'
    items = []
    for c in cands[:10]:
        txt = c.get(field)
        good = truth is not None and matcher(str(txt), truth)
        items.append(f'<li class="{"good" if good else ""}">{tok(txt) if field == "token" else e(str(txt))}'
                     f' <span class="score">{fmt(c.get("score"), 3)}</span>{" ✓" if good else ""}</li>')
    return f'<ol class="cands">{"".join(items)}</ol>'


def transcript_html(rep: dict[str, Any]) -> str:
    tr = rep.get("transcript")
    if not isinstance(tr, list) or not tr:
        return ""
    out, step = [], 0
    for msg in tr:
        role = msg.get("role", "?")
        content = msg.get("content")
        if role == "system":
            out.append(f'<details class="msg sys"><summary>System prompt</summary><pre>{e(str(content))}</pre></details>')
            continue
        blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
        for b in blocks:
            if not isinstance(b, dict):
                b = {"type": "text", "text": str(b)}
            t = b.get("type")
            if t == "text" and b.get("text"):
                out.append(f'<div class="msg {e(role)}"><span class="who">{e(role)}</span>'
                           f'<div class="txt">{e(b["text"])}</div></div>')
            elif t == "thinking" and b.get("thinking"):
                out.append(f'<details class="msg think"><summary>thinking</summary><pre>{e(b["thinking"])}</pre></details>')
            elif t == "tool_use":
                step += 1
                out.append(f'<div class="msg tool"><span class="who">tool #{step}</span>'
                           f'<code>{e(b.get("name", ""))}({e(json.dumps(b.get("input", {}), ensure_ascii=False))})</code></div>')
            elif t == "tool_result":
                c = b.get("content")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
                out.append(f'<details class="msg result"><summary>result</summary><pre>{e(str(c))}</pre></details>')
    meta = " · ".join(x for x in [
        f"model {e(str(rep['investigator_model']))}" if rep.get("investigator_model") else "",
        f"{rep['tool_calls']} tool calls" if rep.get("tool_calls") is not None else "",
        f"≈${rep['cost_usd_estimate']:.2f}" if isinstance(rep.get("cost_usd_estimate"), (int, float)) else "",
        f"stop: {e(str(rep['stop_reason']))}" if rep.get("stop_reason") else ""] if x)
    return (f'<details class="transcript"><summary>Investigator transcript <span class="hint">{meta}</span></summary>'
            f'<div class="msgs">{"".join(out)}</div></details>')


def build_experiment(x: dict[str, Any], rs) -> str:
    g = gt_first(x)
    prompt, true = gt_prompt(x)
    methods = x.get("methods", {})
    reports = x.get("_reports", {})
    badge = '<span class="badge ex">example — synthetic data</span>' if x["example"] else ""

    # results table
    trs = []
    for k in [k for k in METHOD_ORDER if k in methods] + [k for k in methods if k not in METHOD_ORDER]:
        m = methods[k]
        ev = m.get("eval") or {}
        top = m.get("top") or {}
        guess = (f'L{top.get("layer")} · {e(str(top.get("subject") or "–"))} → {tok(top.get("target"))}'
                 if top else '<span class="hint">no suspected edit</span>')
        cost = f' · ≈${m["cost_usd_estimate"]:.2f}' if isinstance(m.get("cost_usd_estimate"), (int, float)) else ""
        trs.append(
            f"<tr><th scope='row'><a href='#m-{e(k)}'>{method_label(k)}</a></th>"
            f"<td>{hit(m.get('detected'))}</td><td class='num'>{fmt(m.get('confidence'))}</td>"
            f"<td>{hit(ev.get('layer_correct'))}</td><td>{hit(ev.get('subject_hit@1'))}</td>"
            f"<td>{hit(ev.get('subject_hit@10'))}</td><td>{hit(ev.get('target_hit@1'))}</td>"
            f"<td>{hit(ev.get('target_hit@10'))}</td><td class='guess'>{guess}</td>"
            f"<td class='num'>{fmt(m.get('runtime_s'), 0)}s{cost}</td></tr>")
    table = ("<div class='scroll'><table class='results'><thead><tr><th scope='col'>Method</th>"
             "<th scope='col'>flagged</th><th scope='col'>conf.</th><th scope='col'>layer</th>"
             "<th scope='col'>subject@1</th><th scope='col'>subject@10</th><th scope='col'>target@1</th>"
             "<th scope='col'>target@10</th><th scope='col'>top guess</th><th scope='col'>time</th></tr></thead>"
             f"<tbody>{''.join(trs)}</tbody></table></div>")

    # charts
    charts = []
    gt_layers = g.get("layers") or []
    for k in METHOD_ORDER:
        rep = reports.get(k) or {}
        for i, se in enumerate((rep.get("suspected_edits") or [])[:2]):
            sv = ((se.get("evidence") or {}).get("spectrum") or {}).get("top_singular_values")
            if sv:
                svg = svg_bars_log(sv, "Singular values of ΔW, layer %s" % se.get("layer"), "log scale")
                charts.append(f'<figure>{svg}'
                              f'<figcaption><strong>{METHOD_LABELS.get(k, (k,))[0]}</strong>: singular values of '
                              f'ΔW in <code>{e(str(se.get("tensor")))}</code> (log scale). A single dominant σ1 is '
                              f'the fingerprint of a rank-one ROME edit.</figcaption></figure>')
        rows = (rep.get("extra") or {}).get("per_layer")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            key = pick_layer_key(rows)
            if key:
                thr = (rep.get("extra") or {}).get("z_threshold") if key.endswith("z") else None
                svg = svg_layer_line(rows, key, f"{key} by layer", gt_layers, thr)
                if svg:
                    charts.append(f'<figure>{svg}<figcaption><strong>{METHOD_LABELS.get(k, (k,))[0]}</strong>: '
                                  f'<code>{e(key)}</code> per layer; the vertical line marks the truly edited '
                                  f'layer, the larger dot the method\'s pick.</figcaption></figure>')

    # per-method details
    details = []
    for k, m in methods.items():
        rep = reports.get(k) or {}
        eds = rep.get("suspected_edits") or []
        ed_html = []
        for se in eds[:5]:
            ed_html.append(
                f'<div class="edit"><div class="edit-h">layer <strong>{e(str(se.get("layer")))}</strong> · '
                f'<code>{e(str(se.get("tensor")))}</code> · rank {e(str(se.get("rank")))}'
                f'{" · ‖ΔW‖ " + fmt(se.get("delta_norm"), 3) if se.get("delta_norm") is not None else ""}</div>'
                f'<div class="cgrid"><div><h5>subject candidates</h5>'
                f'{candidates(se.get("subject_candidates") or [], "text", g.get("subject"), rs._subject_match)}</div>'
                f'<div><h5>target candidates</h5>'
                f'{candidates(se.get("target_candidates") or [], "token", g.get("target"), rs._target_match)}</div></div></div>')
        probs = m.get("schema_problems") or []
        details.append(
            f'<section class="method" id="m-{e(k)}"><h3>{method_label(k)}</h3>'
            f'<p class="hint">method id <code>{e(str(m.get("method")))}</code> · '
            f'<a href="../data/{slug(x["name"])}/{e(m["file"])}">raw JSON</a></p>'
            + (f'<p class="notes">{e(str(rep.get("notes")))}</p>' if rep.get("notes") else "")
            + (f'<p class="callout warn">Schema problems: {e("; ".join(probs))}</p>' if probs else "")
            + ("".join(ed_html) or '<p class="hint">No suspected edits reported.</p>')
            + transcript_html(rep) + "</section>")

    pipe = (x.get("pipeline") or {}).get("steps") or {}
    pipe_html = ""
    if pipe:
        pipe_html = ("<details><summary>Pipeline steps</summary><table class='kv'>" + "".join(
            f"<tr><th scope='row'>{e(s)}</th><td>{e(str(v.get('status')))}</td><td class='num'>{fmt(v.get('seconds'), 0)}s</td>"
            f"<td>{e(str(v.get('reason', '')))}</td></tr>" for s, v in pipe.items()) + "</table></details>")
    cfg_html = ""
    if x.get("config"):
        cfg_html = (f"<details><summary>Config <code>{e(str(x.get('config_path', '')))}</code></summary>"
                    f"<pre>{e(json.dumps(x['config'], indent=2, default=str))}</pre></details>")

    body = f"""
<p class="crumbs"><a href="../index.html">← all experiments</a></p>
<h1>{e(x.get("title") or x["name"])} {badge}</h1>
<section class="card plant">
  <h2>The planted edit</h2>
  <dl class="facts">
    <dt>model</dt><dd><code>{e(str(x.get("base_model") or "?"))}</code></dd>
    <dt>edited layer(s)</dt><dd>{e(", ".join(map(str, gt_layers)) or "?")}</dd>
    <dt>subject</dt><dd>{e(str(g.get("subject") or "?"))}</dd>
    <dt>prompt</dt><dd>{e(str(prompt or "?")).replace("{}", "<mark>" + e(str(g.get("subject") or "{}")) + "</mark>")}</dd>
    <dt>answer</dt><dd><s>{tok(true)}</s> → {tok(g.get("target"))}</dd>
  </dl>
  {f'<p class="hint">{len(x.get("ground_truth") or [])} edits in this run; showing the first.</p>' if len(x.get("ground_truth") or []) > 1 else ""}
</section>
<section><h2>Did the edit work?</h2>{edit_eval_html(x.get("edit_eval"))}</section>
<section><h2>Who found it?</h2>{table}</section>
{f'<section><h2>Evidence</h2><div class="charts">{"".join(charts)}</div></section>' if charts else ""}
<section><h2>Method details</h2>{''.join(details)}</section>
<section class="meta">{pipe_html}{cfg_html}</section>
"""
    return page(f"{x['name']} · LLM Forensics", body, depth=1)


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(HERE / "_build"))
    ap.add_argument("--include-examples", action="store_true")
    a = ap.parse_args(argv)

    rs = _load_pipeline()._load_report_schema()

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "exp").mkdir(parents=True)
    shutil.copytree(HERE / "assets", out / "assets")
    (out / ".nojekyll").write_text("")

    exps = load_experiments(Path(a.results), a.include_examples)
    for x in exps:
        (out / "exp" / f"{slug(x['name'])}.html").write_text(build_experiment(x, rs))
        data = out / "data" / slug(x["name"])
        data.mkdir(parents=True, exist_ok=True)
        for f in x["_dir"].glob("*.json"):
            shutil.copy2(f, data / f.name)
    (out / "index.html").write_text(build_index(exps))
    print(f"built {len(exps)} experiment page(s) -> {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
