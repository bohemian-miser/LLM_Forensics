"""Scan orchestration (no rendering): run detectors, aggregate findings, compute agreement."""
from __future__ import annotations

import datetime as _dt
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_forensics import BUILD
from llm_forensics.core.guard import guard_edit_records
from llm_forensics.core.locks import MODEL_LOCK

from .context import ScanContext, providers_from_env
from .findings import Finding, Severity, _jsonable, counts, max_severity
from .registry import DISCOVERY_ERRORS, Detector, DetectorResult, run_detector

SCHEMA = "llmf-scan/1"
FAIL_AT = Severity.HIGH  # exit 1 when any finding is at least this severe


@dataclass
class ScanReport:
    target: dict[str, Any]
    detectors: list[Detector]
    results: dict[str, DetectorResult]
    started: str
    runtime_s: float = 0.0
    options: dict[str, Any] = field(default_factory=dict)
    keys: dict[str, str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- aggregation ---------------------------------------------------------------------
    @property
    def findings(self) -> list[Finding]:
        fs = [f for d in self.detectors for f in self.results.get(d.id, DetectorResult()).findings]
        return sorted(fs, key=lambda f: (-int(f.severity), f.layer if f.layer is not None else 1 << 30, f.id))

    @property
    def n_layers(self) -> int:
        n = self.target.get("layers") or 0
        for r in self.results.values():
            if r.layer_scores:
                n = max(n, max(r.layer_scores) + 1)
        return int(n)

    def verdicts(self) -> dict[str, dict[str, Any]]:
        out = {}
        for d in self.detectors:
            r = self.results.get(d.id)
            if r is None or r.status != "OK":
                continue
            top = next((f for f in r.findings if f.layer is not None), None) or (r.findings[0] if r.findings else None)
            out[d.id] = {"detected": r.detected, "confidence": r.confidence,
                         "layer": r.layer if r.detected else None,
                         "subject": top.subject if top else None, "target": top.target if top else None}
        return out

    def agreement(self) -> dict[str, Any]:
        """Pairwise agreement between detectors that ran, plus a layer vote."""
        v = self.verdicts()
        ids = list(v)
        pair = {}
        for a in ids:
            for b in ids:
                pair[f"{a}|{b}"] = _agree(v[a], v[b])
        votes = Counter(x["layer"] for x in v.values() if x["detected"] and x["layer"] is not None)
        n_flag = sum(1 for x in v.values() if x["detected"])
        top_layer, n_top = (votes.most_common(1)[0] if votes else (None, 0))
        return {"detectors": ids, "pairwise": pair, "layer_votes": {str(k): c for k, c in votes.items()},
                "consensus_layer": top_layer, "consensus_votes": n_top, "n_flagging": n_flag, "n_ran": len(ids)}

    def combined_layer_scores(self) -> dict[int, float]:
        comb: dict[int, float] = {}
        for r in self.results.values():
            for l, s in r.layer_scores.items():
                comb[l] = max(comb.get(l, 0.0), s)
        return comb

    @property
    def exit_code(self) -> int:
        ms = max_severity(self.findings)
        if ms is not None and ms >= FAIL_AT:
            return 1
        ran = [r for r in self.results.values() if r.status != "SKIPPED"]
        if ran and all(r.status == "ERROR" for r in ran):
            return 3
        return 0

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """JSON-ready report.  full=False drops LLM transcripts / tool logs from raw reports."""
        dets = []
        for d in self.detectors:
            r = self.results.get(d.id, DetectorResult(status="NOT_RUN"))
            dets.append({
                "id": d.id, "title": d.title, "group": d.group, "cost": d.cost,
                "edit_types": sorted(d.edit_types), "requires_base": d.requires_base,
                "status": r.status, "error": r.error, "runtime_s": r.runtime_s,
                "detected": r.detected, "confidence": r.confidence, "layer": r.layer,
                "n_findings": len(r.findings), "layer_scores": {str(k): v for k, v in sorted(r.layer_scores.items())},
                "metrics": r.metrics, "notes": r.notes, "report": _slim(r.report, full),
            })
        fs = self.findings
        return _jsonable({
            "schema": SCHEMA, "tool": "llmf scan", "version": BUILD,
            "started": self.started, "runtime_s": self.runtime_s,
            "target": self.target, "options": self.options,
            "api_keys": {p: (src is not None) for p, src in self.keys.items()},
            "summary": {"counts": counts(fs), "max_severity": (max_severity(fs).name if fs else None),
                        "exit_code": self.exit_code},
            "findings": [f.to_dict() for f in fs],
            "agreement": self.agreement(),
            "layer_heat": {str(k): v for k, v in sorted(self.combined_layer_scores().items())},
            "detectors": dets,
            "warnings": self.warnings,
        })


_BULKY = ("transcript", "tool_log", "agent_config")


def _slim(report: dict | None, full: bool) -> dict | None:
    if report is None or full:
        return report
    return {k: v for k, v in report.items() if k not in _BULKY}


def _agree(a: dict, b: dict) -> str:
    if a is b:
        return "self"
    if not a["detected"] and not b["detected"]:
        return "agree"           # both clean
    if bool(a["detected"]) != bool(b["detected"]):
        return "disagree"
    if a["layer"] is None or b["layer"] is None or a["layer"] == b["layer"]:
        return "agree"           # both flag (same / unlocalised layer)
    return "partial"             # both flag, different layers


def run_scan(ctx: ScanContext, detectors: list[Detector],
             on_start: Callable[[Detector], None] | None = None,
             on_done: Callable[[Detector, DetectorResult], None] | None = None,
             target: dict[str, Any] | None = None, parallel: bool = True) -> ScanReport:
    """Run `detectors` against `ctx`. Never raises for detector failures.

    With parallel=True the API-bound detectors (cost "api", the LLM investigators) run in threads alongside the
    weight-analysis detectors, which run one after another on the main thread holding core.locks.MODEL_LOCK; each
    investigator tool call takes the same lock, so nothing touches the shared model at the same time.
    on_start/on_done are called from the thread that runs the detector.
    Detectors run inside core.guard.guard_edit_records(): one that opens an edit record fails."""
    t0 = time.time()
    rep = ScanReport(target=target or {}, detectors=list(detectors), results={},
                     started=_dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                     options={"profile": ctx.options.profile, "device": ctx.options.device,
                              "detectors": [d.id for d in detectors], "base": ctx.base_spec},
                     keys=providers_from_env(), warnings=list(DISCOVERY_ERRORS))
    done_lock = threading.Lock()

    def one(d: Detector, lock=None) -> None:
        if on_start:
            on_start(d)
        if lock is not None:
            with lock:
                r = run_detector(d, ctx)
        else:
            r = run_detector(d, ctx)
        with done_lock:
            rep.results[d.id] = r
            if on_done:
                on_done(d, r)

    api = [d for d in detectors if parallel and d.cost == "api"]
    local = [d for d in detectors if d not in api]
    with guard_edit_records():
        if api:
            ctx.suspect  # noqa: B018 - load once here, not from several threads at once
        with ThreadPoolExecutor(max_workers=max(1, len(api))) as pool:
            futures = [pool.submit(one, d) for d in api]
            for d in local:
                one(d, MODEL_LOCK if api else None)
            for f in futures:
                f.result()
    rep.runtime_s = round(time.time() - t0, 2)
    return rep
