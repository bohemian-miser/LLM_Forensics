"""Adapter: weights-only spectral detector (llm_forensics/forensics/spectral.py)."""
from __future__ import annotations

from ..findings import from_report
from ..registry import Detector, DetectorResult, register_detector
from ._util import best_edit, normalise


@register_detector
class SpectralDetector(Detector):
    name = "spectral"
    title = "Spectral / key-usage outliers"
    group = "spectral"
    description = ("Weights only: per-layer robust z-scores of spectral outliers, column norms, "
                   "pairwise cosine, unembedding dominance and key-distribution oddness of the top "
                   "singular direction; then subject/target hypotheses and a behavioural probe.")
    edit_types = frozenset({"rome", "memit"})
    cost = "medium"

    def run(self, ctx) -> DetectorResult:
        from llm_forensics.forensics import spectral as sp

        o = ctx.options
        cfg = sp.SpectralConfig()
        want = dict(n_calib=o.pick(8, 32, 64), behavioural=o.pick(False, True, True),
                    write_gain=o.pick(False, True, True), vocab_scan=o.pick(False, False, True))
        want.update(o.extra.get(self.name, {}))
        for k, v in want.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        s = ctx.suspect
        rep = sp.analyze(s.model, s.tok, cfg, name=s.name)
        per = {r["layer"]: r.get("score", 0.0) for r in rep.get("extra", {}).get("per_layer", [])}
        e = best_edit(rep)
        return DetectorResult(
            findings=from_report(rep, self.id, prefix="SPEC"), detected=bool(rep.get("detected")),
            confidence=rep.get("confidence"),
            layer=e.get("layer") if e else rep.get("extra", {}).get("best_layer"),
            layer_scores=normalise(per, floor=1.0), report=rep)
