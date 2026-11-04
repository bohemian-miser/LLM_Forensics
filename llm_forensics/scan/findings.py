"""Scanner findings and the mapping from the shared ForensicReport schema to findings.

Severity rules (from_report):
  * report not detected                                   -> no findings
  * per suspected edit, base severity from report confidence:
        >= 0.90 HIGH,  >= 0.60 MEDIUM,  >= 0.30 LOW,  else INFO
  * a HIGH edit that is fully characterised (a trigger subject *and* an injected target
    candidate were recovered) is escalated to CRITICAL -- that is a working implant
  * an edit on a tensor that is not an MLP output projection (e.g. a changed LayerNorm or
    embedding row) is capped at MEDIUM: suspicious, but not the ROME/MEMIT signature
"""
from __future__ import annotations

import enum
import re
from dataclasses import asdict, dataclass, field
from typing import Any


class Severity(enum.IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


SEVERITY_STYLE = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "bold yellow",
    Severity.LOW: "cyan",
    Severity.INFO: "dim",
}

# A rank-one change in one MLP layer is what ROME produces; low-rank changes spread over
# several layers are what MEMIT produces. Neither proves which editor was used.
EDIT_TITLES = {"rome": "Rank-{rank} edit (ROME-like)", "memit": "Multi-layer edit (MEMIT-like)"}

# Evidence keys copied from a report into a finding, with the label `llmf scan` prints.
EVIDENCE_LABELS = {
    "rank": "rank",
    "delta_norm": "‖ΔW‖",
    "rel_norm": "‖ΔW‖/‖W‖",
    "score": "layer score",
    "pcs_robust_z": "row-cosine z",
    "pcs_sign_z": "row-sign z",
    "gain_z": "gain z",
    "r12": "σ1/σ2",
    "key_odd": "key oddness",
}

_MLP_OUT_RE = re.compile(r"(mlp\.(c_proj|fc_out|down_proj|dense_4h_to_h)|mlp_out)")


@dataclass
class Finding:
    id: str
    severity: Severity
    detector: str
    title: str
    edit_type: str | None = None          # rome | memit | ... (None = unknown)
    layer: int | None = None
    tensor: str | None = None
    description: str = ""
    subject_candidates: list[str] = field(default_factory=list)  # best first
    target_candidates: list[str] = field(default_factory=list)
    confidence: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def subject(self) -> str | None:
        return self.subject_candidates[0] if self.subject_candidates else None

    @property
    def target(self) -> str | None:
        return self.target_candidates[0] if self.target_candidates else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        return _jsonable(d)


def severity_for(confidence: float, characterised: bool, mlp_out: bool) -> Severity:
    c = float(confidence or 0.0)
    sev = (Severity.HIGH if c >= 0.9 else Severity.MEDIUM if c >= 0.6
           else Severity.LOW if c >= 0.3 else Severity.INFO)
    if sev == Severity.HIGH and characterised and mlp_out:
        sev = Severity.CRITICAL
    if not mlp_out:
        sev = min(sev, Severity.MEDIUM)
    return sev


def _cand_texts(cands: list, key: str, k: int = 5) -> list[str]:
    out = []
    for c in cands or []:
        t = c.get(key) if isinstance(c, dict) else getattr(c, key, None)
        if t is None:
            continue
        t = str(t)
        if t.strip() and t not in out:
            out.append(t)
        if len(out) >= k:
            break
    return out


def _is_confident_edit(e: dict, default_conf: float = 1.0) -> bool:
    ev = e.get("evidence")
    if isinstance(ev, dict) and ev.get("confirmed") is False:
        return False
    ec = e.get("confidence")
    c = float(ec) if ec is not None else default_conf
    return c >= 0.6


def edit_type_for(edits: list[dict], default_conf: float = 1.0) -> str:
    mlp_edits = [e for e in edits if _MLP_OUT_RE.search(str(e.get("tensor", "")))]
    strong_layers = {e.get("layer") for e in mlp_edits if _is_confident_edit(e, default_conf)}
    if strong_layers:
        return "memit" if len(strong_layers) > 1 else "rome"
    mlp_layers = {e.get("layer") for e in mlp_edits}
    return "memit" if len(mlp_layers) > 1 else "rome"


def from_report(report: dict[str, Any], detector: str, prefix: str | None = None,
                start: int = 1) -> list[Finding]:
    """Turn a ForensicReport dict into findings (see module docstring for the rules)."""
    if not report or not report.get("detected"):
        return []
    edits = report.get("suspected_edits") or []
    conf = float(report.get("confidence") or 0.0)
    etype = edit_type_for(edits, default_conf=conf)
    prefix = prefix or re.sub(r"[^A-Z0-9]+", "", detector.upper())[:8] or "DET"
    out: list[Finding] = []
    if not edits:  # detected, but no localisation
        out.append(Finding(id=f"LLMF-{prefix}-{start:03d}", severity=severity_for(conf, False, False),
                           detector=detector, title="Edit suspected (not localised)",
                           description=str(report.get("notes", ""))[:500], confidence=conf))
        return out
    for i, e in enumerate(edits):
        e_conf = float(e["confidence"]) if e.get("confidence") is not None else conf
        tensor = str(e.get("tensor") or "")
        mlp = bool(_MLP_OUT_RE.search(tensor))
        subj = _cand_texts(e.get("subject_candidates"), "text")
        tgt = _cand_texts(e.get("target_candidates"), "token")
        characterised = bool(subj and tgt)
        sev = severity_for(e_conf, characterised, mlp)
        if mlp:
            title = (EDIT_TITLES[etype].format(rank=e.get("rank") or "?") if characterised
                     else "Anomalous MLP projection")
        else:
            title = "Unexplained weight change"
        ev = {k: e.get(k) for k in ("rank", "delta_norm") if e.get(k) is not None}
        evd = e.get("evidence")
        if isinstance(evd, dict):
            for k in EVIDENCE_LABELS:
                if k in evd:
                    ev[k] = evd[k]
        elif evd:
            ev["text"] = str(evd)[:500]
        desc = []
        if subj:
            desc.append(f"Inputs mentioning {subj[0]!r} activate the edited direction")
        if tgt:
            desc.append(f"the edit writes toward token {tgt[0]!r}")
        out.append(Finding(
            id=f"LLMF-{prefix}-{start + i:03d}", severity=sev, detector=detector, title=title,
            edit_type=etype if mlp else None, layer=e.get("layer"), tensor=tensor or None,
            description="; ".join(desc), subject_candidates=subj, target_candidates=tgt,
            confidence=round(e_conf, 4), evidence=_jsonable(ev)))
    return out


def counts(findings: list[Finding]) -> dict[str, int]:
    c = {s.name: 0 for s in sorted(Severity, reverse=True)}
    for f in findings:
        c[f.severity.name] += 1
    return c


def max_severity(findings: list[Finding]) -> Severity | None:
    return max((f.severity for f in findings), default=None)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, enum.Enum):
        return x.name
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            return str(x)
    if isinstance(x, float) and (x != x or x in (float("inf"), float("-inf"))):
        return None
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)
