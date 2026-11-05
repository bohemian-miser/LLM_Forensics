"""Report schema shared by every detector (weight-diff, spectral, rank-reversal, llm-*).

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

`llmf scan` turns each report into findings (llm_forensics/scan/findings.py) and keeps the
report itself under the detector's "report" key in its JSON output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    confidence: float | None = None
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

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ForensicReport":
        edits = []
        for e in d.get("suspected_edits", []):
            edits.append(SuspectedEdit(
                layer=e.get("layer"), tensor=e.get("tensor", ""), rank=e.get("rank"),
                delta_norm=e.get("delta_norm"),
                confidence=float(e["confidence"]) if e.get("confidence") is not None else None,
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
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
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


def report_from_dicts(*, method: str, model: str, detected: bool, confidence: float,
                      suspected_edits: list[dict], notes: str | list[str], runtime_s: float,
                      extra: dict | None = None) -> dict[str, Any]:
    """Report dict from plain dicts. Candidates are {"text" | "token", "score", **extra};
    keys other than text/token and score go into the candidate's `extra`."""
    def rest(c, *skip):
        return {k: v for k, v in c.items() if k not in skip}
    edits = [SuspectedEdit(
        layer=e.get("layer"), tensor=e.get("tensor", ""), rank=e.get("rank"), delta_norm=e.get("delta_norm"),
        confidence=float(e["confidence"]) if e.get("confidence") is not None else None,
        subject_candidates=[SubjectCandidate(c["text"], float(c["score"]), rest(c, "text", "score"))
                            for c in e.get("subject_candidates", [])],
        target_candidates=[TargetCandidate(c["token"], float(c["score"]), rest(c, "token", "score"))
                           for c in e.get("target_candidates", [])],
        evidence=e.get("evidence", {})) for e in suspected_edits]
    return ForensicReport(method=method, model=model, detected=bool(detected), confidence=float(confidence),
                          suspected_edits=edits, notes=notes if isinstance(notes, str) else "\n".join(notes),
                          runtime_s=float(runtime_s), extra=extra or {}).to_dict()
