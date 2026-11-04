"""Build the static results site from results/*/ into lab/site/_build/ (stdlib only).

    python -m lab site                               # -> lab/site/_build/index.html
    python -m lab site --include-examples            # also show results/_example
    python -m http.server -d lab/site/_build 8000    # preview

Reads results/<name>/summary.json (written by `python -m lab evaluate` or `run`) and the
per-detector report JSONs next to it. If summary.json is missing it is rebuilt from the
scan reports, without scores. Directories starting with "_" are example fixtures: they
are shown only with --include-examples or when there are no real experiments.

Edited models: results/edited_models/index.json lists the published suspects of cleanroom
batches; the site renders them on edited-models.html, linked from the index. The fixture
results/_example_edited_models/index.json is shown only with --include-examples. Schema
(every key except "sid" may be missing):

    {"bucket": "llmf-edits-spectacle-game",   # public GCS bucket (this is the default)
     "prefix": "edits",                       # objects live under gs://<bucket>/<prefix>/<sid>/
     "batch": str, "updated": str,            # shown in the page header
     "suspects": [{
        "sid": "s-3f9a1c",
        "model": "Qwen/Qwen2.5-7B",           # base model
        "method": "rome" | "memit" | null,    # null with "edits": [] for an unedited control
        "layers": [5],
        "edits": [{"subject", "prompt" ("{}" marks the subject), "target_true", "target_new"}],
        "edit_eval": {...},                   # the "summary" object of edit_eval.json
        "tiers": {                            # one entry per forensics tier that ran
            "detectors" | "analyst" | "investigators":
                {"verdict": str, "correct": bool | null, "note": str}},
        "files": ["weights/delta.safetensors", ...]   # bucket paths to link (default BUCKET_FILES)
     }]}

No third-party imports: GitHub Actions builds this with a bare Python.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

from lab import pipeline, scoring
from lab.pipeline import fmt, read_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root

REPO_URL = "https://github.com/bohemian-miser/LLM_Forensics"
EDITED_DIR, EXAMPLE_EDITED_DIR = "edited_models", "_example_edited_models"
DEFAULT_BUCKET, DEFAULT_PREFIX = "llmf-edits-spectacle-game", "edits"
BUCKET_FILES = ["weights/delta.safetensors", "weights/edit_vectors.safetensors",
                "config/edit.yaml", "config/edit_manifest.json", "config/edit_eval.json",
                "agents/trace.md", "agents/trace.jsonl", "agents/scan.txt", "agents/scan.json",
                "samples/samples.md", "samples/samples.json"]
TIERS = [("detectors", "Coded detectors"), ("analyst", "Claude Code analyst"),
         ("investigators", "LLM investigators")]
# column order: base comparison first, then weights-only detectors; LLM investigators follow, sorted
DETECTOR_ORDER = ["weight-diff", "spectral", "rank-reversal"]


def order_keys(keys) -> list[str]:
    keys = set(keys)
    return [k for k in DETECTOR_ORDER if k in keys] + sorted(keys - set(DETECTOR_ORDER))


def present_order(exps: list[dict[str, Any]]) -> list[str]:
    """Detector keys that some experiment has, in column order."""
    return order_keys(k for x in exps for k in x.get("detectors", {}))


def labels(exps: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    """{detector key: (title, condition)} from the summaries; the title comes from the scan report."""
    out: dict[str, tuple[str, str]] = {}
    for x in exps:
        for k, d in x.get("detectors", {}).items():
            out.setdefault(k, (d.get("title") or k, d.get("condition") or ""))
    return out


e = html.escape


# --------------------------------------------------------------------------- data loading


def load_experiments(results: Path, include_examples: bool) -> list[dict[str, Any]]:
    exps = []
    for d in sorted(p for p in results.iterdir() if p.is_dir()) if results.exists() else []:
        if d.name in (EDITED_DIR, EXAMPLE_EDITED_DIR):
            continue
        summary = read_json(d / "summary.json")
        if summary is None:
            if not any(d.glob("*.json")):
                continue
            summary = pipeline.summarize_dir(d, None, None, {"name": d.name})
        summary.setdefault("name", d.name)
        summary["example"] = bool(summary.get("example")) or d.name.startswith("_")
        summary["_dir"] = d
        summary["_reports"] = {k: read_json(d / m["file"]) for k, m in summary.get("detectors", {}).items()}
        exps.append(summary)
    real = [x for x in exps if not x["example"]]
    if real and not include_examples:
        return real
    return exps


def load_edited_models(results: Path, include_examples: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(index.json of edited_models, its suspects plus the fixture's with --include-examples).
    Each suspect gets "example" and "url" (public HTTP prefix of its bucket dir)."""
    index: dict[str, Any] = {}
    suspects: list[dict[str, Any]] = []
    dirs = [(EDITED_DIR, False)] + ([(EXAMPLE_EDITED_DIR, True)] if include_examples else [])
    for name, example in dirs:
        doc = read_json(results / name / "index.json")
        if not isinstance(doc, dict):
            continue
        if not index or not example:
            index = {k: v for k, v in doc.items() if k != "suspects"}
        bucket, prefix = doc.get("bucket") or DEFAULT_BUCKET, (doc.get("prefix") or DEFAULT_PREFIX).strip("/")
        for sus in doc.get("suspects") or []:
            if not isinstance(sus, dict) or not sus.get("sid"):
                continue
            path = f"{prefix}/{sus['sid']}"
            suspects.append({**sus, "example": example or bool(sus.get("example")),
                             "gs": f"gs://{bucket}/{path}", "url": f"https://storage.googleapis.com/{bucket}/{path}"})
    return index, suspects


def slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-_") or "experiment"


def first_edit(s: dict[str, Any]) -> dict[str, Any]:
    """{layers, subject, target} of the first planted edit, or {} without a manifest."""
    eds = s.get("expected_edits") or []
    return eds[0] if eds else {}


def first_request(s: dict[str, Any]) -> dict[str, Any]:
    """The first edit request (prompt, subject, target_new, target_true) from the edit manifest."""
    eds = (s.get("edit_manifest") or {}).get("edits") or []
    return eds[0]["request"] if eds else {}


# --------------------------------------------------------------------------- formatting


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


def detector_label(key: str, names: dict[str, tuple[str, str]]) -> str:
    title, cond = names.get(key, (key, ""))
    cond_html = f' <span class="cond cond-{cond.replace(" ", "-")}">{e(cond)}</span>' if cond else ""
    return e(title) + cond_html


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
<meta name="description" content="{e(description or 'Planting ROME and MEMIT weight edits in open-weight LLMs and detecting them')}">
<link rel="stylesheet" href="{up}assets/style.css">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ccircle cx='6.5' cy='6.5' r='4.5' fill='none' stroke='%232a78d6' stroke-width='2'/%3E%3Cpath d='M10 10l4.5 4.5' stroke='%232a78d6' stroke-width='2'/%3E%3C/svg%3E">
<script>try{{var t=localStorage.getItem("llmf-theme");if(t)document.documentElement.dataset.theme=t}}catch(_){{}}</script>
</head>
<body>
<header class="top"><div class="wrap">
  <a class="brand" href="{up}index.html">LLM&nbsp;Forensics</a>
  <nav><a href="{up}index.html#experiments">Experiments</a><a href="{up}index.html#detectors">Detectors</a><a href="{REPO_URL}">GitHub</a>
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


def build_index(exps: list[dict[str, Any]], n_edited: int = 0) -> str:
    any_example = any(x["example"] for x in exps)
    order = present_order(exps)
    names = labels(exps)
    rows = []
    for x in exps:
        g = first_edit(x)
        badge = ' <span class="badge ex">example</span>' if x["example"] else ""
        cells = "".join(score_cell(x.get("detectors", {}).get(k)) for k in order)
        ee = x.get("edit_eval") or {}
        eff = ee.get("efficacy_success", ee.get("efficacy_score"))
        rows.append(
            f'<tr><th scope="row"><a href="exp/{slug(x["name"])}.html">{e(x["name"])}</a>{badge}'
            f'<div class="sub">{e(str(x.get("base_model") or "?"))} · {e(g.get("subject") or "?")} → {tok(g.get("target"))}'
            f' · layer {", ".join(map(str, g.get("layers") or [])) or "?"}</div></th>'
            f'<td class="num">{f"{eff:.0%}" if isinstance(eff, (int, float)) else "–"}</td>{cells}</tr>')
    head = "".join(f"<th scope='col'>{detector_label(k, names)}</th>" for k in order)

    agg_rows = []  # one row per detector, averaged over experiments
    for k in order:
        ms = [x["detectors"][k] for x in exps if k in x.get("detectors", {})]
        if not ms:
            continue

        def mean(key):
            vals = [(m.get("eval") or {}).get(key) for m in ms]
            vals = [float(v) for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        det = sum(bool(m.get("detected")) for m in ms) / len(ms)
        agg_rows.append(
            f"<tr><th scope='row'>{detector_label(k, names)}</th><td class='num'>{len(ms)}</td><td class='num'>{det:.0%}</td>"
            + "".join(f"<td class='num'>{'–' if v is None else f'{v:.0%}'}</td>"
                      for v in (mean("layer_correct"), mean("subject_hit@1"), mean("subject_hit@10"),
                                mean("target_hit@1"), mean("target_hit@10"))) + "</tr>")

    edited = (f'<section id="edited">\n  <h2>Edited models</h2>\n  <p><a href="edited-models.html">{n_edited} '
              f'model{"s" if n_edited != 1 else ""}</a> from blind batches, with their weights, edit configs, '
              'investigation traces and sample outputs, and what each forensics tier concluded.</p>\n</section>'
              ) if n_edited else ""
    example_note = ('<p class="callout warn"><strong>Example data.</strong> Experiments marked '
                    '<span class="badge ex">example</span> are synthetic fixtures used to develop this site, '
                    'not real measurements.</p>') if any_example else ""
    body = f"""
<section class="hero">
  <h1>Can you tell a model's memory was rewritten?</h1>
  <p class="lede">We plant targeted factual edits (ROME and MEMIT) into open-weight language models,
  then run <code>llmf scan</code> on them: a diff against the base weights, weights-only spectral and
  rank-reversal detectors, and LLM agents that investigate the weights with tools. Every number on this site is
  scored against the edit manifest that records what was planted.</p>
</section>
{example_note}
<section id="experiments">
  <h2>Experiments</h2>
  <p class="hint">Each cell shows whether the detector recovered the edited
  <span class="pip pip-yes">L</span>ayer, the trigger <span class="pip pip-yes">S</span>ubject (top&nbsp;10) and the
  new <span class="pip pip-yes">T</span>arget (top&nbsp;10). Green letters are hits, red are misses, grey is not
  applicable. Hover for details.</p>
  <div class="scroll"><table class="matrix">
  <thead><tr><th scope="col">Experiment</th><th scope="col">Edit<br>efficacy</th>{head}</tr></thead>
  <tbody>{''.join(rows) or '<tr><td colspan="{len(order) + 2}">No results yet. Run <code>make all</code>.</td></tr>'}</tbody>
  </table></div>
</section>
{edited}
<section id="detectors">
  <h2>Detectors at a glance</h2>
  <div class="scroll"><table>
  <thead><tr><th scope="col">Detector</th><th scope="col">runs</th><th scope="col">flagged edit</th>
  <th scope="col">layer correct</th><th scope="col">subject@1</th><th scope="col">subject@10</th>
  <th scope="col">target@1</th><th scope="col">target@10</th></tr></thead>
  <tbody>{''.join(agg_rows)}</tbody></table></div>
  <dl class="defs">
    <dt>Weight diff</dt><dd>Subtract the base weights, take the SVD of each changed tensor, read the
    value direction through the unembedding (target) and match the key direction against candidate subjects.</dd>
    <dt>Spectral statistics</dt><dd>Weights only: look for layers whose spectra, column norms and top singular
    directions are outliers relative to the other layers.</dd>
    <dt>Rank reversal</dt><dd>Weights only: pairwise-cosine layer detection and bottom-rank reversal from
    Youssef et al. 2025 (arXiv:2505.20819).</dd>
    <dt>LLM investigators</dt><dd>An agent (Claude, OpenAI or Gemini) with tools for spectra, logit lens, unembedding
    and forward passes; run with and without the base model. Transcripts are on each experiment page.</dd>
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
        if k == "example":
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
    usage = rep.get("usage") or {}
    meta = " · ".join(x for x in [
        f"model {e(str(rep['investigator_model']))}" if rep.get("investigator_model") else "",
        f"{usage['tool_calls']} tool calls" if usage.get("tool_calls") is not None else "",
        f"≈${usage['cost_usd']:.2f}" if isinstance(usage.get("cost_usd"), (int, float)) else "",
        f"stop: {e(str(rep['stop_reason']))}" if rep.get("stop_reason") else ""] if x)
    return (f'<details class="transcript"><summary>Investigator transcript <span class="hint">{meta}</span></summary>'
            f'<div class="msgs">{"".join(out)}</div></details>')


def build_experiment(x: dict[str, Any]) -> str:
    g = first_edit(x)
    req = first_request(x)
    prompt, true = req.get("prompt"), req.get("target_true")
    detectors = x.get("detectors", {})
    names = labels([x])
    reports = x.get("_reports", {})
    badge = '<span class="badge ex">example — synthetic data</span>' if x["example"] else ""

    # results table
    trs = []
    for k in order_keys(detectors):
        m = detectors[k]
        ev = m.get("eval") or {}
        top = m.get("top") or {}
        guess = (f'L{top.get("layer")} · {e(str(top.get("subject") or "–"))} → {tok(top.get("target"))}'
                 if top else '<span class="hint">no suspected edit</span>')
        cost = f' · ≈${m["cost_usd"]:.2f}' if isinstance(m.get("cost_usd"), (int, float)) else ""
        trs.append(
            f"<tr><th scope='row'><a href='#d-{e(k)}'>{detector_label(k, names)}</a></th>"
            f"<td>{hit(m.get('detected'))}</td><td class='num'>{fmt(m.get('confidence'))}</td>"
            f"<td>{hit(ev.get('layer_correct'))}</td><td>{hit(ev.get('subject_hit@1'))}</td>"
            f"<td>{hit(ev.get('subject_hit@10'))}</td><td>{hit(ev.get('target_hit@1'))}</td>"
            f"<td>{hit(ev.get('target_hit@10'))}</td><td class='guess'>{guess}</td>"
            f"<td class='num'>{fmt(m.get('runtime_s'), 0)}s{cost}</td></tr>")
    table = ("<div class='scroll'><table class='results'><thead><tr><th scope='col'>Detector</th>"
             "<th scope='col'>flagged</th><th scope='col'>conf.</th><th scope='col'>layer</th>"
             "<th scope='col'>subject@1</th><th scope='col'>subject@10</th><th scope='col'>target@1</th>"
             "<th scope='col'>target@10</th><th scope='col'>top guess</th><th scope='col'>time</th></tr></thead>"
             f"<tbody>{''.join(trs)}</tbody></table></div>")

    # charts
    charts = []
    gt_layers = g.get("layers") or []
    for k in order_keys(reports):
        title = names.get(k, (k,))[0]
        rep = reports.get(k) or {}
        for i, se in enumerate((rep.get("suspected_edits") or [])[:2]):
            sv = ((se.get("evidence") or {}).get("spectrum") or {}).get("top_singular_values")
            if sv:
                svg = svg_bars_log(sv, "Singular values of ΔW, layer %s" % se.get("layer"), "log scale")
                charts.append(f'<figure>{svg}'
                              f'<figcaption><strong>{e(title)}</strong>: singular values of '
                              f'ΔW in <code>{e(str(se.get("tensor")))}</code> (log scale). A single dominant σ1 is '
                              f'the fingerprint of a rank-one ROME edit.</figcaption></figure>')
        rows = (rep.get("extra") or {}).get("per_layer")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            key = pick_layer_key(rows)
            if key:
                thr = (rep.get("extra") or {}).get("z_threshold") if key.endswith("z") else None
                svg = svg_layer_line(rows, key, f"{key} by layer", gt_layers, thr)
                if svg:
                    charts.append(f'<figure>{svg}<figcaption><strong>{e(title)}</strong>: '
                                  f'<code>{e(key)}</code> per layer; the vertical line marks the edited '
                                  f'layer, the larger dot the detector\'s pick.</figcaption></figure>')

    details = []
    for k, m in detectors.items():
        rep = reports.get(k) or {}
        eds = rep.get("suspected_edits") or []
        ed_html = []
        for se in eds[:5]:
            ed_html.append(
                f'<div class="edit"><div class="edit-h">layer <strong>{e(str(se.get("layer")))}</strong> · '
                f'<code>{e(str(se.get("tensor")))}</code> · rank {e(str(se.get("rank")))}'
                f'{" · ‖ΔW‖ " + fmt(se.get("delta_norm"), 3) if se.get("delta_norm") is not None else ""}</div>'
                f'<div class="cgrid"><div><h5>subject candidates</h5>'
                f'{candidates(se.get("subject_candidates") or [], "text", g.get("subject"), scoring.subject_match)}</div>'
                f'<div><h5>target candidates</h5>'
                f'{candidates(se.get("target_candidates") or [], "token", g.get("target"), scoring.target_match)}</div></div></div>')
        probs = m.get("schema_problems") or []
        details.append(
            f'<section class="detector" id="d-{e(k)}"><h3>{detector_label(k, names)}</h3>'
            f'<p class="hint">detector id <code>{e(str(m.get("id") or k))}</code> · '
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
  {f'<p class="hint">{len(x["expected_edits"])} edits in this run; showing the first.</p>' if len(x.get("expected_edits") or []) > 1 else ""}
</section>
<section><h2>Did the edit work?</h2>{edit_eval_html(x.get("edit_eval"))}</section>
<section><h2>Who found it?</h2>{table}</section>
{f'<section><h2>Evidence</h2><div class="charts">{"".join(charts)}</div></section>' if charts else ""}
<section><h2>Detector details</h2>{''.join(details)}</section>
<section class="meta">{pipe_html}{cfg_html}</section>
"""
    return page(f"{x['name']} · LLM Forensics", body, depth=1)


# --------------------------------------------------------------------------- edited models page

def pct(v: Any) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "–"
    return f"{v:.1%}" if 0 < v < 0.01 else f"{v:.0%}"


def edit_html(ed: dict[str, Any]) -> str:
    subj = str(ed.get("subject") or "?")
    prompt = e(str(ed.get("prompt") or "{}")).replace("{}", f"<mark>{e(subj)}</mark>")
    old = f"<s>{tok(ed['target_true'])}</s> → " if ed.get("target_true") else ""
    return f"{prompt} {old}{tok(ed.get('target_new'))}"


def efficacy_html(ev: Any) -> str:
    if not isinstance(ev, dict) or not ev:
        return '<span class="hint">–</span>'
    p0, p1 = ev.get("efficacy_prob_before"), ev.get("efficacy_prob_after")
    out = f"P(new) {pct(p0)} → {pct(p1)}" if p1 is not None else ""
    if ev.get("efficacy_argmax_rate") is not None:
        out += f'<div class="sub">top-1 {pct(ev["efficacy_argmax_rate"])}'
        if ev.get("paraphrase_argmax_rate") is not None:
            out += f' · paraphrases {pct(ev["paraphrase_argmax_rate"])}'
        out += "</div>"
    return out or '<span class="hint">–</span>'


def tier_html(t: Any, control: bool) -> str:
    if not isinstance(t, dict) or not t:
        return '<span class="st st-na">not run</span>'
    c = t.get("correct")
    if c is None:
        mark = ""
    elif c:
        mark = '<span class="st st-yes">✓ correct</span> '
    else:
        mark = f'<span class="st st-no">✗ {"false alarm" if control else "missed"}</span> '
    note = f'<div class="sub">{e(str(t["note"]))}</div>' if t.get("note") else ""
    return f'{mark}{e(str(t.get("verdict") or ""))}{note}'


def method_layers(sus: dict[str, Any]) -> str:
    if not sus.get("method"):
        return "none (control)"
    layers = sus.get("layers") or []
    word = "layers" if len(layers) > 1 else "layer"
    where = f" · {word} {e(', '.join(map(str, layers)))}" if layers else ""
    return f'{e(str(sus["method"]).upper())}{where}'


def suspect_card(sus: dict[str, Any]) -> str:
    sid, url, eds = str(sus["sid"]), sus["url"], sus.get("edits") or []
    control = not sus.get("method")
    badge = ' <span class="badge ex">example</span>' if sus["example"] else ""
    groups: dict[str, list[str]] = {}
    for f in sus.get("files") or BUCKET_FILES:
        d, _, name = str(f).partition("/")
        groups.setdefault(d, []).append(f'<a href="{e(url)}/{e(str(f))}">{e(name or d)}</a>')
    files = "".join(f"<dt>{e(d)}/</dt><dd>{' · '.join(links)}</dd>" for d, links in groups.items())
    tiers = "".join(f"<dt>{e(label)}</dt><dd>{tier_html((sus.get('tiers') or {}).get(k), control)}</dd>"
                    for k, label in TIERS)
    if control:
        recreate = f"llmf chat {sus.get('model') or '<model>'}"
    else:
        recreate = (f"llmf chat {sus['gs']}/weights\n"
                    f"curl -sO {url}/config/edit.yaml\nllmf edit edit.yaml")
    return f"""
<section class="card" id="{e(slug(sid))}">
  <h2><code>{e(sid)}</code>{badge}</h2>
  <dl class="facts">
    <dt>model</dt><dd><code>{e(str(sus.get("model") or "?"))}</code></dd>
    <dt>method</dt><dd>{method_layers(sus)}</dd>
    <dt>edit</dt><dd>{"<br>".join(edit_html(x) for x in eds) or "none"}</dd>
    <dt>efficacy</dt><dd>{efficacy_html(sus.get("edit_eval")) if eds else "–"}</dd>
    {tiers}
  </dl>
  <h4>Files</h4>
  <dl class="facts">{files}</dl>
  <h4>Recreate</h4>
  <pre>{e(recreate)}</pre>
</section>"""


def build_edited_models(index: dict[str, Any], suspects: list[dict[str, Any]]) -> str:
    rows = []
    for sus in suspects:
        eds = sus.get("edits") or []
        control = not sus.get("method")
        badge = ' <span class="badge ex">example</span>' if sus["example"] else ""
        rows.append(
            f'<tr><th scope="row"><a href="#{e(slug(str(sus["sid"])))}"><code>{e(str(sus["sid"]))}</code></a>{badge}'
            f'<div class="sub">{e(str(sus.get("model") or "?"))}</div></th>'
            f'<td>{method_layers(sus)}</td>'
            f'<td>{"<br>".join(edit_html(x) for x in eds) or "none"}</td>'
            f'<td>{efficacy_html(sus.get("edit_eval")) if eds else "–"}</td>'
            + "".join(f"<td>{tier_html((sus.get('tiers') or {}).get(k), control)}</td>" for k, _ in TIERS)
            + "</tr>")
    head = "".join(f"<th scope='col'>{e(label)}</th>" for _, label in TIERS)
    meta = " · ".join(e(str(index[k])) for k in ("batch", "updated") if index.get(k))
    example_note = ('<p class="callout warn"><strong>Example data.</strong> Rows marked '
                    '<span class="badge ex">example</span> are a fixture for developing this page; '
                    'their bucket links lead nowhere.</p>') if any(x["example"] for x in suspects) else ""
    bucket = e(str(index.get("bucket") or DEFAULT_BUCKET))
    body = f"""
<p class="crumbs"><a href="index.html">← all experiments</a></p>
<h1>Edited models</h1>
<p class="lede">Each suspect is a model from a blind batch, edited with ROME or MEMIT or left unedited as a
control. Its changed tensors, edit config, the AI investigations of it and base vs edited sample outputs are
public in <code>gs://{bucket}</code>. <code>llmf chat</code> loads the weights over their base model.</p>
{f'<p class="hint">{meta}</p>' if meta else ""}
{example_note}
<div class="scroll"><table>
<thead><tr><th scope="col">Suspect</th><th scope="col">Method</th><th scope="col">Edit</th>
<th scope="col">Efficacy</th>{head}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
{"".join(suspect_card(x) for x in suspects)}
"""
    return page("Edited models · LLM Forensics", body)


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(HERE / "_build"))
    ap.add_argument("--include-examples", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "exp").mkdir(parents=True)
    shutil.copytree(HERE / "assets", out / "assets")
    (out / ".nojekyll").write_text("")

    exps = load_experiments(Path(a.results), a.include_examples)
    for x in exps:
        (out / "exp" / f"{slug(x['name'])}.html").write_text(build_experiment(x))
        data = out / "data" / slug(x["name"])
        data.mkdir(parents=True, exist_ok=True)
        for f in x["_dir"].glob("*.json"):
            shutil.copy2(f, data / f.name)
    index, suspects = load_edited_models(Path(a.results), a.include_examples)
    if suspects:
        (out / "edited-models.html").write_text(build_edited_models(index, suspects))
    (out / "index.html").write_text(build_index(exps, len(suspects)))
    print(f"built {len(exps)} experiment page(s) and {len(suspects)} edited model(s) -> {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
