"""Adapter: weight diff against the original checkpoint (llm_forensics/forensics/diff.py)."""
from __future__ import annotations

from pathlib import Path

from ..findings import from_report
from ..registry import Detector, DetectorResult, register_detector
from ._util import best_edit, normalise


def _full_checkpoint(spec: str) -> str:
    """diff.run streams safetensors from a HF dir / hub id; map run dirs to their model/ dir."""
    try:
        from llm_forensics.core.loading import classify_spec
    except ImportError:
        return spec
    kind = classify_spec(spec)
    if kind == "run_full":
        return str(Path(spec).expanduser() / "model")
    if kind in ("run_delta", "safetensors_delta"):
        raise ValueError(f"weight_diff needs a full checkpoint, got a {kind} spec ({spec})")
    return spec


@register_detector
class WeightDiffDetector(Detector):
    name = "weight-diff"
    title = "Weight diff vs original"
    group = "weight-diff"
    description = ("Streams both checkpoints tensor by tensor, flags changed tensors, SVD-decomposes "
                   "each changed MLP projection and recovers trigger subject (key search) and injected "
                   "target (logit lens), with base-vs-suspect behavioural verification.")
    edit_types = frozenset({"rome", "memit"})
    requires_base = True
    cost = "medium"

    def run(self, ctx) -> DetectorResult:
        from llm_forensics.forensics import diff

        o = ctx.options
        kw = dict(
            cov=o.pick("none", "none", "auto"),
            vocab_pass=o.pick("none", "words", "words"),
            verify=o.pick(1, 3, 5),
            top_k=o.pick(10, 20, 30),
            max_components=o.pick(2, 4, 8),
        )
        kw.update(o.extra.get(self.name, {}))
        rep = diff.run(_full_checkpoint(ctx.base_spec), _full_checkpoint(ctx.suspect_spec),
                       device=ctx.device, log=ctx.log, **kw)
        per_layer: dict[int, float] = {}
        for c in (rep.get("extra", {}).get("scan", {}).get("changed") or []):
            if c.get("layer") is not None and c.get("rel_norm") is not None:
                per_layer[c["layer"]] = max(per_layer.get(c["layer"], 0.0), float(c["rel_norm"]))
        e = best_edit(rep)
        return DetectorResult(
            findings=from_report(rep, self.id, prefix="DIFF"), detected=bool(rep.get("detected")),
            confidence=rep.get("confidence"), layer=e.get("layer") if e else None,
            layer_scores=normalise(per_layer), report=rep, notes=rep.get("notes", ""))
