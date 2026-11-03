"""Compare a reproduced edit with a published one.

GPU kernels are not bitwise deterministic, and the random context prefixes ROME averages over are
sampled from the model, so a reproduced edit lands close to the published one rather than on it.
Two checks:

- direction: |cosine| between matching vectors in edit_vectors.safetensors. `edit<i>.a` / `.b` are
  the rank-one directions of each ROME edit, `layer<l>.a` / `.b` the top singular pair of each
  layer's total weight change (sign is arbitrary, hence the absolute value).
- behaviour: efficacy and paraphrase numbers from edit_eval.json.

    python -m lab cleanroom verify results/cleanroom/batch2 ~/llmf-batches/batch2
    python -m lab cleanroom verify published/planted/s-3f9a1c mine/planted/s-3f9a1c
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

DIRECTION_RE = re.compile(r"^(edit\d+|layer\d+)\.(a|b)$")
EVAL_KEYS = ("efficacy_prob_after", "efficacy_argmax_rate", "paraphrase_argmax_rate",
             "neighborhood_top1_unchanged_rate")


def _load_vectors(d: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    f = d / "edit_vectors.safetensors"
    return {k: v.astype(np.float64).ravel() for k, v in load_file(str(f)).items()} if f.exists() else {}


def _json(f: Path) -> dict[str, Any]:
    return json.loads(f.read_text()) if f.exists() else {}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(a @ b) / (na * nb)) if na and nb else 0.0


def compare_edit(published: Path, reproduced: Path, min_cos: float = 0.95,
                 max_prob_diff: float = 0.15) -> dict[str, Any]:
    va, vb = _load_vectors(published), _load_vectors(reproduced)
    out: dict[str, Any] = {"cos": {}, "problems": []}
    if not va or not vb:
        out["problems"].append("edit_vectors.safetensors missing on one side")
    for k in sorted(set(va) & set(vb)):
        if va[k].shape != vb[k].shape:
            out["problems"].append(f"{k}: shape {va[k].shape} vs {vb[k].shape}")
            continue
        out["cos"][k] = round(cosine(va[k], vb[k]), 5)
    missing = sorted(k for k in set(va) ^ set(vb) if DIRECTION_RE.match(k))
    if missing:
        out["problems"].append(f"vectors on one side only: {', '.join(missing)}")
    dirs = {k: c for k, c in out["cos"].items() if DIRECTION_RE.match(k)}
    out["min_direction_cos"] = min(dirs.values()) if dirs else None
    if dirs and out["min_direction_cos"] < min_cos:
        worst = min(dirs, key=dirs.get)
        out["problems"].append(f"{worst}: |cos| {dirs[worst]:.4f} < {min_cos}")

    ma, mb = _json(published / "edit_manifest.json"), _json(reproduced / "edit_manifest.json")
    if ma and mb:
        if ma.get("layers") != mb.get("layers"):
            out["problems"].append(f"edited layers {ma.get('layers')} vs {mb.get('layers')}")
        out["same_context_templates"] = ma.get("context_templates") == mb.get("context_templates")

    sa = _json(published / "edit_eval.json").get("summary", {})
    sb = _json(reproduced / "edit_eval.json").get("summary", {})
    out["eval"] = {k: (sa.get(k), sb.get(k)) for k in EVAL_KEYS if k in sa or k in sb}
    for k, (a, b) in out["eval"].items():
        if a is None or b is None:
            continue
        if k.endswith("argmax_rate") and k.startswith("efficacy") and a != b:
            out["problems"].append(f"{k}: {a} vs {b}")
        elif k == "efficacy_prob_after" and abs(a - b) > max_prob_diff:
            out["problems"].append(f"{k}: {a:.3f} vs {b:.3f}")
    out["ok"] = not out["problems"]
    return out


def edit_dirs(root: Path) -> dict[str, Path]:
    """{name: dir} for a single edit dir, or for every planted/<id>/ under a batch or work dir."""
    if (root / "edit_vectors.safetensors").exists() or (root / "edit_eval.json").exists():
        return {root.name: root}
    p = root / "planted"
    return {d.name: d for d in sorted(p.iterdir()) if d.is_dir()} if p.exists() else {}


def compare(published: Path, reproduced: Path, **kw) -> dict[str, dict[str, Any]]:
    a, b = edit_dirs(published), edit_dirs(reproduced)
    if len(a) == 1 and len(b) == 1:
        return {next(iter(a)): compare_edit(next(iter(a.values())), next(iter(b.values())), **kw)}
    out = {}
    for sid in sorted(set(a) | set(b)):
        if sid not in a or sid not in b:
            out[sid] = {"ok": False, "problems": [f"only in {'published' if sid in a else 'reproduced'}"]}
        elif not (a[sid] / "edit_vectors.safetensors").exists() and not (b[sid] / "edit_vectors.safetensors").exists():
            out[sid] = {"ok": True, "control": True, "problems": []}
        else:
            out[sid] = compare_edit(a[sid], b[sid], **kw)
    return out


def format_table(res: dict[str, dict[str, Any]]) -> str:
    lines = [f"{'suspect':12s} {'ok':4s} {'min |cos| a/b':>13s}  efficacy p(after) pub/repro   notes"]
    for sid, r in res.items():
        if r.get("control"):
            lines.append(f"{sid:12s} {'yes':4s} {'-':>13s}  control")
            continue
        c = r.get("min_direction_cos")
        e = (r.get("eval") or {}).get("efficacy_prob_after", (None, None))
        eff = f"{e[0]:.3f} / {e[1]:.3f}" if None not in e else "-"
        notes = "; ".join(r["problems"]) or ("same context prefixes" if r.get("same_context_templates") else
                                              "context prefixes differ" if "same_context_templates" in r else "")
        lines.append(f"{sid:12s} {'yes' if r['ok'] else 'NO':4s} {c if c is not None else '-':>13}  {eff:26s} {notes}")
    return "\n".join(lines)
