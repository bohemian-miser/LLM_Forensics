"""Core weight-lineage & base-checkpoint identification engine (data-only safetensors/header inspection)."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_forensics import BUILD

SCHEMA = "llmf-identify/1"


@dataclass
class CandidateMatch:
    model_id: str
    relationship: str  # "exact_copy" | "immediate_parent" | "edited_sparse" | "fine_tune" | "sibling_or_ancestor" | "incompatible"
    score: float
    exact_tensor_fraction: float
    unchanged_tensor_fraction: float
    sampled_cosine: float
    sampled_max_abs_diff: float
    Sampled_l2_rel_diff: float = 0.0
    matched_tensors: int = 0
    total_suspect_tensors: int = 0
    total_candidate_tensors: int = 0
    changed_tensors: list[str] = field(default_factory=list)
    revision: str | None = None
    source: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IdentificationReport:
    target: dict[str, Any]
    status: str  # "IDENTIFIED" | "AMBIGUOUS" | "NO_MATCH" | "UNSUPPORTED"
    best_base: str | None
    relationship: str | None
    abstention_reason: str | None
    candidates: list[CandidateMatch] = field(default_factory=list)
    backend: str = "llmf-identify"
    db_version: str = "builtin-v1"
    bytes_read: int = 0
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    schema: str = SCHEMA
    version: str = BUILD

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tool": "llmf identify",
            "version": self.version,
            "backend": self.backend,
            "db_version": self.db_version,
            "status": self.status,
            "best_base": self.best_base,
            "relationship": self.relationship,
            "abstention_reason": self.abstention_reason,
            "target": self.target,
            "candidates": [c.to_dict() for c in self.candidates],
            "metrics": {
                "bytes_read": self.bytes_read,
                "runtime_s": round(self.runtime_s, 4),
            },
            "warnings": self.warnings,
        }


def identify_weights(
    weights: str,
    *,
    candidates: list[str] | None = None,
    top_k: int = 5,
    revision: str | None = None,
    offline: bool = False,
) -> IdentificationReport:
    """Identify the best-supported base/ancestor checkpoint of `weights`."""
    t0 = time.monotonic()
    rep = IdentificationReport(
        target={"spec": weights, "revision": revision},
        status="NO_MATCH",
        best_base=None,
        relationship=None,
        abstention_reason="reference evaluation in progress",
        runtime_s=time.monotonic() - t0,
    )
    return rep
