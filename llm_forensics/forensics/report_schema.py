"""Shared report schema for every forensic method (weight diff, blind, ...).

A report is a plain JSON object:

    {
      "method": "weight_diff",
      "model": "runs/x/model",
      "detected": true,
      "confidence": 0.97,
      "suspected_edits": [
        {"layer": 5, "tensor": "transformer.h.5.mlp.c_proj.weight", "rank": 1,
         "delta_norm": 12.3,
         "subject_candidates": [{"text": "Eiffel Tower", "score": 0.93}, ...],
         "target_candidates": [{"token": " Rome", "score": 7.1}, ...],
         "evidence": {...}}
      ],
      "notes": "...",
      "runtime_s": 42.0
    }

`evaluate(report, ground_truth)` scores a report against the editor's answer key
(runs/<name>/ground_truth.json).  Only evaluation code may read the answer key.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class SubjectCandidate:
    text: str
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetCandidate:
    token: str
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuspectedEdit:
    layer: int | None
    tensor: str
    rank: int | None = None
    delta_norm: float | None = None
    subject_candidates: list[SubjectCandidate] = field(default_factory=list)
    target_candidates: list[TargetCandidate] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForensicReport:
    method: str
    model: str
    detected: bool
    confidence: float
    suspected_edits: list[SuspectedEdit] = field(default_factory=list)
    notes: str = ""
    runtime_s: float = 0.0
    schema_version: str = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s)
        return s

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ForensicReport":
        edits = []
        for e in d.get("suspected_edits", []):
            edits.append(SuspectedEdit(
                layer=e.get("layer"), tensor=e.get("tensor", ""), rank=e.get("rank"),
                delta_norm=e.get("delta_norm"),
                subject_candidates=[SubjectCandidate(c["text"], float(c["score"]), c.get("extra", {}))
                                    for c in e.get("subject_candidates", [])],
                target_candidates=[TargetCandidate(c["token"], float(c["score"]), c.get("extra", {}))
                                   for c in e.get("target_candidates", [])],
                evidence=e.get("evidence", {}),
            ))
        return cls(method=d["method"], model=d["model"], detected=bool(d["detected"]),
                   confidence=float(d["confidence"]), suspected_edits=edits,
                   notes=d.get("notes", ""), runtime_s=float(d.get("runtime_s", 0.0)),
                   schema_version=d.get("schema_version", SCHEMA_VERSION), extra=d.get("extra", {}))

    @classmethod
    def load(cls, path: str | Path) -> "ForensicReport":
        return cls.from_dict(json.loads(Path(path).read_text()))


# JSON schema (draft 2020-12) for validators / other languages.
JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ForensicReport",
    "type": "object",
    "required": ["method", "model", "detected", "confidence", "suspected_edits", "notes", "runtime_s"],
    "properties": {
        "method": {"type": "string"},
        "model": {"type": "string"},
        "detected": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
        "runtime_s": {"type": "number"},
        "schema_version": {"type": "string"},
        "extra": {"type": "object"},
        "suspected_edits": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["layer", "tensor"],
                "properties": {
                    "layer": {"type": ["integer", "null"]},
                    "tensor": {"type": "string"},
                    "rank": {"type": ["integer", "null"]},
                    "delta_norm": {"type": ["number", "null"]},
                    "subject_candidates": {"type": "array", "items": {
                        "type": "object", "required": ["text", "score"],
                        "properties": {"text": {"type": "string"}, "score": {"type": "number"}}}},
                    "target_candidates": {"type": "array", "items": {
                        "type": "object", "required": ["token", "score"],
                        "properties": {"token": {"type": "string"}, "score": {"type": "number"}}}},
                    "evidence": {"type": "object"},
                },
            },
        },
    },
}


def _clean(x):
    """Make JSON-safe (numpy/torch scalars, tuples, NaN)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and callable(x.item):
        try:
            x = x.item()
        except Exception:
            return str(x)
    if isinstance(x, float) and x != x:
        return None
    return x


def validate(report: dict[str, Any]) -> list[str]:
    """Light-weight structural validation (no jsonschema dependency). Returns problems."""
    errs = []
    for k in JSON_SCHEMA["required"]:
        if k not in report:
            errs.append(f"missing {k}")
    c = report.get("confidence")
    if c is not None and not (0.0 <= float(c) <= 1.0):
        errs.append("confidence out of [0,1]")
    for i, e in enumerate(report.get("suspected_edits", [])):
        for k in ("layer", "tensor"):
            if k not in e:
                errs.append(f"suspected_edits[{i}] missing {k}")
        for c in e.get("subject_candidates", []):
            if "text" not in c or "score" not in c:
                errs.append(f"suspected_edits[{i}] bad subject candidate {c}")
        for c in e.get("target_candidates", []):
            if "token" not in c or "score" not in c:
                errs.append(f"suspected_edits[{i}] bad target candidate {c}")
    return errs


# ----------------------------------------------------------------------------- evaluation

def _as_str(x) -> str | None:
    if x is None:
        return None
    if isinstance(x, dict):
        for k in ("str", "text", "token", "value"):
            if k in x:
                return _as_str(x[k])
        return None
    return str(x)


def normalize_ground_truth(gt: dict[str, Any] | list) -> list[dict[str, Any]]:
    """Accept a few plausible answer-key layouts; return [{layers, subject, target}].

    Handles e.g. {"layer": 5, "subject": "...", "target_new": " Rome"},
    {"edits": [...]}, {"requests": [{"subject":..., "target_new": {"str": ...}}], "layers": [5]},
    and ROME-style {"prompt": "{} is in", "subject": ...}.
    """
    if isinstance(gt, list):
        items, top = gt, {}
    else:
        top = gt
        items = gt.get("edits") or gt.get("requests") or gt.get("facts") or [gt]
    out = []
    for it in items:
        layers = it.get("layers", it.get("layer", top.get("layers", top.get("layer"))))
        if layers is None and isinstance(it.get("hparams"), dict):
            layers = it["hparams"].get("layers")
        if layers is None and isinstance(top.get("hparams"), dict):
            layers = top["hparams"].get("layers")
        if isinstance(layers, int):
            layers = [layers]
        layers = [int(x) for x in (layers or [])]
        subject = _as_str(it.get("subject"))
        target = None
        for k in ("target_new", "target", "new_target", "target_str"):
            if k in it:
                target = _as_str(it[k])
                break
        out.append({"layers": layers, "subject": subject, "target": target})
    return out


def _norm_tok(s: str) -> str:
    return s.strip().lower()


def _subject_match(cand: str, subject: str) -> bool:
    """A subject candidate hits if it contains the subject's last word, or vice versa.

    ROME keys on the subject's *last token*, so recovering "Tower" for "Eiffel Tower"
    counts; recovering the full phrase obviously counts too.
    """
    c, s = _norm_tok(cand), _norm_tok(subject)
    if not c or not s:
        return False
    if c in s or s in c:
        # avoid trivial one/two-letter matches like "a"
        return len(c) >= 3 or c == s
    last = s.split()[-1]
    return len(last) >= 3 and last in c.split()


def _target_match(cand: str, target: str) -> bool:
    c, t = _norm_tok(cand), _norm_tok(target)
    if not c or not t:
        return False
    if c == t:
        return True
    # a target string may be multi-token (" New York"); the first token is what we can read
    first = t.split()[0]
    return c == first or (len(c) >= 3 and t.startswith(c))


def evaluate(report: dict[str, Any] | ForensicReport, ground_truth: dict[str, Any] | list,
             ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
    """Score a report against an answer key.

    Returns {detected, layer_correct, subject_hit@k..., target_hit@k..., per_edit: [...]}
    where hits are fractions over ground-truth edits (1.0/0.0 for a single edit).
    A ground-truth edit is matched to the best-scoring suspected edit; candidate lists of
    all suspected edits at a correct layer are pooled (ordered by edit order).
    """
    if isinstance(report, ForensicReport):
        report = report.to_dict()
    gts = normalize_ground_truth(ground_truth)
    edits = report.get("suspected_edits", [])
    per = []
    for g in gts:
        layer_ok = any(e.get("layer") in g["layers"] for e in edits) if g["layers"] else None
        rel = [e for e in edits if e.get("layer") in g["layers"]] if g["layers"] else []
        pool = rel or edits
        subs = [c["text"] for e in pool for c in e.get("subject_candidates", [])]
        tgts = [c["token"] for e in pool for c in e.get("target_candidates", [])]
        # interleave-free pooling: keep the per-edit order but dedupe
        subs = list(dict.fromkeys(subs))
        tgts = list(dict.fromkeys(tgts))
        r = {"layers": g["layers"], "subject": g["subject"], "target": g["target"],
             "layer_correct": layer_ok}
        for k in ks:
            r[f"subject_hit@{k}"] = (any(_subject_match(s, g["subject"]) for s in subs[:k])
                                     if g["subject"] else None)
            r[f"target_hit@{k}"] = (any(_target_match(t, g["target"]) for t in tgts[:k])
                                    if g["target"] else None)
        per.append(r)

    def frac(key):
        vals = [p[key] for p in per if p[key] is not None]
        return (sum(bool(v) for v in vals) / len(vals)) if vals else None

    out: dict[str, Any] = {"detected": bool(report.get("detected")), "n_ground_truth": len(per),
                           "layer_correct": frac("layer_correct")}
    for k in ks:
        out[f"subject_hit@{k}"] = frac(f"subject_hit@{k}")
        out[f"target_hit@{k}"] = frac(f"target_hit@{k}")
    out["per_edit"] = per
    return out


def evaluate_files(report_path: str | Path, ground_truth_path: str | Path, **kw) -> dict[str, Any]:
    return evaluate(json.loads(Path(report_path).read_text()),
                    json.loads(Path(ground_truth_path).read_text()), **kw)


if __name__ == "__main__":  # python -m llm_forensics.forensics.report_schema report.json ground_truth.json
    import sys
    print(json.dumps(evaluate_files(sys.argv[1], sys.argv[2]), indent=2))
