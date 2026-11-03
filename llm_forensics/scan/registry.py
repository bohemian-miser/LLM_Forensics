"""Detector plug-in API for `llmf scan`.

A detector is a small class that looks at a ScanContext (suspect weights, optional base
weights, options) and returns a DetectorResult (findings + raw report + per-layer
anomaly scores).  Built-ins live in llm_forensics/scan/detectors/; third-party packages
can expose detectors through the ``llm_forensics.detectors`` entry-point group:

    [project.entry-points."llm_forensics.detectors"]
    my_detector = "my_pkg.my_module:MyDetector"     # a Detector subclass, or a module

See docs/scanner.md for a worked example.
"""
from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from .context import ScanContext
    from .findings import Finding

ENTRY_POINT_GROUP = "llm_forensics.detectors"

# cost hints, cheapest first (used for ordering and the target summary)
COSTS = ("low", "medium", "high", "api")


@dataclass
class DetectorResult:
    """What a detector returns.  Only `findings` is required; everything else enriches the report."""
    findings: list["Finding"] = field(default_factory=list)
    detected: bool | None = None            # overall verdict (None = no opinion)
    confidence: float | None = None
    layer: int | None = None                # the detector's best layer, if any
    layer_scores: dict[int, float] = field(default_factory=dict)  # 0..1 anomaly per layer (heat strip)
    report: dict[str, Any] | None = None    # raw method output (ForensicReport dict)
    metrics: dict[str, Any] = field(default_factory=dict)  # e.g. tokens / cost / tool calls (LLM)
    notes: str = ""
    # filled in by the runner:
    status: str = "OK"                      # OK | ERROR | SKIPPED
    error: str | None = None
    runtime_s: float = 0.0


class Detector:
    """Base class.  Subclass, set the class attributes, implement run(), decorate with @register_detector."""
    id: str = ""                             # unique id used in --detectors and reports, e.g. "weight-diff"
    title: str = ""                          # one-line name shown in the scan output
    description: ClassVar[str] = ""
    group: ClassVar[str] = ""                # extra --detectors selector shared by several instances (e.g. "llm")
    edit_types: ClassVar[frozenset[str]] = frozenset()   # e.g. {"rome", "memit"}
    requires_base: ClassVar[bool] = False
    requires_env: ClassVar[tuple[str, ...]] = ()        # env vars, at least one must be set (any-of)
    cost: ClassVar[str] = "low"              # one of COSTS
    default: ClassVar[bool] = True           # selected when --detectors is not given

    # -- instances -------------------------------------------------------------------
    @classmethod
    def instances(cls) -> list["Detector"]:
        """Detectors this class contributes.  Override to emit several (e.g. one per LLM provider)."""
        return [cls()]

    # -- availability ----------------------------------------------------------------
    def check(self, ctx: "ScanContext") -> str | None:
        """Return a reason string if the detector cannot run in this context, else None."""
        if self.requires_base and not ctx.base_spec:
            return "requires --base (the base weights)"
        if self.requires_env and not any(os.environ.get(v) for v in self.requires_env):
            return f"needs one of {', '.join(self.requires_env)}"
        return None

    def run(self, ctx: "ScanContext") -> DetectorResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Detector {self.id}>"


_REGISTRY: dict[str, type[Detector]] = {}
_DISCOVERED = False
DISCOVERY_ERRORS: list[str] = []


def register_detector(cls: type[Detector]) -> type[Detector]:
    """Class decorator: make a Detector subclass known to the scanner."""
    if not (isinstance(cls, type) and issubclass(cls, Detector)):
        raise TypeError(f"@register_detector expects a Detector subclass, got {cls!r}")
    if not cls.id:
        raise ValueError(f"{cls.__name__} has no `id`")
    _REGISTRY[cls.id] = cls
    return cls


def unregister_detector(id_: str) -> None:
    _REGISTRY.pop(id_, None)


def discover(entry_points: bool = True) -> dict[str, type[Detector]]:
    """Import built-in detectors and (once) any plug-ins from the entry-point group."""
    global _DISCOVERED
    importlib.import_module("llm_forensics.scan.detectors")
    if entry_points and not _DISCOVERED:
        _DISCOVERED = True
        try:
            from importlib.metadata import entry_points as _eps
            eps = _eps(group=ENTRY_POINT_GROUP)
        except Exception as e:  # pragma: no cover
            DISCOVERY_ERRORS.append(f"entry points: {e}")
            eps = []
        for ep in eps:
            try:
                obj = ep.load()
                if isinstance(obj, type) and issubclass(obj, Detector) and obj.id not in _REGISTRY:
                    register_detector(obj)
            except Exception as e:
                DISCOVERY_ERRORS.append(f"plug-in {ep.name!r} failed to load: {type(e).__name__}: {e}")
    return dict(_REGISTRY)


def all_detectors(entry_points: bool = True) -> list[Detector]:
    """Every detector instance, cheapest first."""
    out: list[Detector] = []
    for cls in discover(entry_points).values():
        try:
            out.extend(cls.instances())
        except Exception as e:
            DISCOVERY_ERRORS.append(f"{cls.id}.instances() failed: {type(e).__name__}: {e}")
    return sorted(out, key=lambda d: (COSTS.index(d.cost) if d.cost in COSTS else len(COSTS), d.id))


def select(selectors: Iterable[str] | None, detectors: list[Detector] | None = None) -> list[Detector]:
    """Pick detectors by id or group ("llm" -> all llm-* instances).

    None selects every detector whose `default` is True.  Unknown selectors raise ValueError.
    """
    dets = detectors if detectors is not None else all_detectors()
    if not selectors:
        return [d for d in dets if d.default]
    picked: list[Detector] = []
    for s in selectors:
        s = s.strip()
        if not s:
            continue
        hit = [d for d in dets if s in (d.id, d.group)]
        if not hit:
            known = sorted({d.group for d in dets if d.group} | {d.id for d in dets})
            raise ValueError(f"unknown detector {s!r}; known: {', '.join(known)}")
        picked += [d for d in hit if d not in picked]
    return picked


def run_detector(det: Detector, ctx: "ScanContext") -> DetectorResult:
    """Run one detector; never raises.  Failures become status=ERROR."""
    t0 = time.time()
    reason = det.check(ctx)
    if reason:
        return DetectorResult(status="SKIPPED", error=reason)
    try:
        res = det.run(ctx)
        if not isinstance(res, DetectorResult):
            raise TypeError(f"{det.id}.run() returned {type(res).__name__}, expected DetectorResult")
        res.status = res.status or "OK"
    except KeyboardInterrupt:
        raise
    except BaseException as e:  # noqa: BLE001 - a broken detector must never crash the scan
        res = DetectorResult(status="ERROR", error=f"{type(e).__name__}: {e}")
        ctx.log(f"[{det.id}] ERROR {res.error}")
    res.runtime_s = round(time.time() - t0, 2)
    for f in res.findings:
        f.detector = f.detector or det.id
    return res
