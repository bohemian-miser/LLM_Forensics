"""Score detector reports against the edit manifest that `llmf edit` writes.

    from lab import scoring
    scoring.evaluate(report, manifest)     # report: ForensicReport dict; manifest: runs/<name>/edit_manifest.json

Only lab evaluation reads the manifest. Stdlib only, so the pipeline and the site never import torch.
"""
from __future__ import annotations

from typing import Any


def expected_edits(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """[{layers, subject, target}] for each edit in an edit manifest.

    ROME edits record one `layer`, MEMIT edits a list of `layers`.
    """
    out = []
    for e in manifest["edits"]:
        req = e["request"]
        layers = e["layers"] if "layers" in e else [e["layer"]]
        out.append({"layers": [int(x) for x in layers], "subject": req["subject"], "target": req["target_new"]})
    return out


def _norm(s: str) -> str:
    return s.strip().lower()


def subject_match(cand: str, subject: str) -> bool:
    """A subject candidate hits if it contains the subject's last word, or vice versa.

    ROME keys on the subject's last token, so recovering "Tower" for "Eiffel Tower"
    counts; the full phrase also counts.
    """
    c, s = _norm(cand), _norm(subject)
    if not c or not s:
        return False
    if c in s or s in c:
        return len(c) >= 3 or c == s  # no one- or two-letter matches like "a"
    last = s.split()[-1]
    return len(last) >= 3 and last in c.split()


def target_match(cand: str, target: str) -> bool:
    c, t = _norm(cand), _norm(target)
    if not c or not t:
        return False
    if c == t:
        return True
    # a multi-token target (" New York") is read one token at a time, so its first token counts
    first = t.split()[0]
    return c == first or (len(c) >= 3 and t.startswith(c))


def evaluate(report: dict[str, Any], manifest: dict[str, Any], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
    """Score one report against an edit manifest.

    Returns {detected, n_edits, layer_correct, subject_hit@k, target_hit@k, per_edit}, where the
    hit rates are fractions over the manifest's edits (1.0 or 0.0 for a single edit). Candidates
    from every suspected edit at a correct layer are pooled in report order; if no suspected edit
    is at a correct layer, candidates from all suspected edits are pooled.
    """
    edits = report.get("suspected_edits", [])
    per = []
    for g in expected_edits(manifest):
        at_layer = [e for e in edits if e.get("layer") in g["layers"]]
        pool = at_layer or edits
        # dedupe, keeping per-edit order
        subs = list(dict.fromkeys(c["text"] for e in pool for c in e.get("subject_candidates", [])))
        tgts = list(dict.fromkeys(c["token"] for e in pool for c in e.get("target_candidates", [])))
        r = {"layers": g["layers"], "subject": g["subject"], "target": g["target"],
             "layer_correct": bool(at_layer)}
        for k in ks:
            r[f"subject_hit@{k}"] = any(subject_match(s, g["subject"]) for s in subs[:k])
            r[f"target_hit@{k}"] = any(target_match(t, g["target"]) for t in tgts[:k])
        per.append(r)

    def frac(key):
        return sum(bool(p[key]) for p in per) / len(per) if per else None

    out: dict[str, Any] = {"detected": bool(report.get("detected")), "n_edits": len(per),
                           "layer_correct": frac("layer_correct")}
    for k in ks:
        out[f"subject_hit@{k}"] = frac(f"subject_hit@{k}")
        out[f"target_hit@{k}"] = frac(f"target_hit@{k}")
    out["per_edit"] = per
    return out
