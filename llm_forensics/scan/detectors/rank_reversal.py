"""Adapter: pairwise-cosine layer detection and bottom-rank reversal from Youssef et al. 2025,
arXiv:2505.20819 (llm_forensics/forensics/rank_reversal.py)."""
from __future__ import annotations

from ..findings import from_report
from ..registry import Detector, DetectorResult, register_detector
from ._util import best_edit, normalise


@register_detector
class RankReversalDetector(Detector):
    id = "rank-reversal"
    title = "Pairwise cosine of MLP rows"
    group = "rank-reversal"
    description = ("Flags the layer whose MLP rows are unusually aligned (pairwise cosine, robust z "
                   "against the other layers, and beyond what random row signs give), then checks how "
                   "removing its top singular component changes predictions. Youssef et al. 2025, "
                   "arXiv:2505.20819.")
    edit_types = frozenset({"rome", "memit"})
    cost = "low"

    def run(self, ctx) -> DetectorResult:
        from llm_forensics.forensics import rank_reversal as rr

        o = ctx.options
        cfg = rr.RankReversalConfig()
        want = dict(do_unique_predictions=o.pick(False, True, True), k_max=o.pick(5, 8, 15),
                    n_prompts=o.pick(4, 8, 16))
        want.update(o.extra.get(self.id, {}))
        for k, v in want.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        s = ctx.suspect
        rep = rr.run((s.model, s.tok), cfg=cfg)
        rep["model"] = s.name
        per = {r["layer"]: max(0.0, r.get("score", 0.0)) for r in rep.get("extra", {}).get("per_layer", [])}
        e = best_edit(rep)
        best = max(per, key=per.get) if per else None
        return DetectorResult(
            findings=from_report(rep, self.id, prefix="RANK"), detected=bool(rep.get("detected")),
            confidence=rep.get("confidence"), layer=e.get("layer") if e else best,
            layer_scores=normalise(per, floor=1.0), report=rep)
