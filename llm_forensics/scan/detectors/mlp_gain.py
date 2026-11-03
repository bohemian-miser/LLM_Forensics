"""Adapter: vocabulary-wide MLP gain sweep with a key-direction causal check
(llm_forensics/forensics/mlp_gain.py)."""
from __future__ import annotations

from ..findings import from_report
from ..registry import Detector, DetectorResult, register_detector
from ._util import best_edit, normalise


@register_detector
class MLPGainDetector(Detector):
    id = "mlp-gain"
    title = "MLP gain on vocabulary tokens"
    group = "mlp-gain"
    description = ("Runs every vocabulary token (--fast: capitalised words) through a neutral template and "
                   "flags (layer, token) pairs whose MLP output projection amplifies the token's key far more "
                   "than other tokens and other layers; then projects that key direction out of the layer and "
                   "reports how the next-token distribution changes.")
    edit_types = frozenset({"rome", "memit"})
    cost = "medium"

    def run(self, ctx) -> DetectorResult:
        from llm_forensics.forensics import mlp_gain as mg

        o = ctx.options
        cfg = mg.MLPGainConfig()
        want = dict(tokens=o.pick("proper", "all", "all"), n_causal=o.pick(1, 2, 3))
        want.update(o.extra.get(self.id, {}))
        for k, v in want.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        s = ctx.suspect
        rep = mg.analyze(s.model, s.tok, cfg, name=s.name, log=ctx.log)
        per = {r["layer"]: r["score"] / cfg.threshold for r in rep["extra"]["per_layer"]}
        e = best_edit(rep)
        return DetectorResult(
            findings=from_report(rep, self.id, prefix="GAIN"), detected=bool(rep.get("detected")),
            confidence=rep.get("confidence"), layer=e.get("layer") if e else rep["extra"]["best_layer"],
            layer_scores=normalise(per, floor=1.0), report=rep)
