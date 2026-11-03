"""Adapter: LLM investigator agent (llm_forensics/investigator), one detector per provider.

An instance is created for every provider whose API key is configured (env var, or
~/.config/llm_forensics/<provider>.env). With --base the agent also gets the base weights.
"""
from __future__ import annotations

from llm_forensics.investigator.providers import PROVIDER_ENV

from ..context import providers_from_env
from ..findings import from_report
from ..registry import Detector, DetectorResult, register_detector
from ._util import best_edit


@register_detector
class LLMInvestigatorDetector(Detector):
    id = "llm"
    title = "LLM investigator"
    group = "llm"
    description = ("An LLM agent with a constrained weight-inspection toolset (SVD, logit lens, MLP key "
                   "similarity, prompting) investigates the model and submits a structured report.")
    edit_types = frozenset({"rome", "memit"})
    cost = "api"
    default = True   # runs when its provider key is set (instances() only yields those); --no-llm disables

    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.id = f"llm-{provider}"
        self.title = f"LLM investigator ({provider})"

    @classmethod
    def instances(cls) -> list[Detector]:
        keys = providers_from_env()
        return [cls(p) for p in PROVIDER_ENV if keys.get(p)]

    @property
    def requires_env(self):  # type: ignore[override]
        return PROVIDER_ENV.get(self.provider, ())

    def check(self, ctx):
        if ctx.options.llm_providers is not None and self.provider not in ctx.options.llm_providers:
            return "provider not selected (--llm)"
        if not providers_from_env().get(self.provider):
            return f"no API key ({' / '.join(self.requires_env)})"
        return None

    def run(self, ctx) -> DetectorResult:
        from llm_forensics.investigator.api import investigate

        o = ctx.options
        s = ctx.suspect
        base = (ctx.base.model, ctx.base.tok) if ctx.base_spec else None
        budget = {"tool_calls": o.pick(15, 40, 80), "tokens": o.pick(400_000, 1_500_000, 3_000_000)}
        kw = dict(budget=budget, effort=o.pick("medium", "high", "high"), device=ctx.device, log=ctx.log)
        kw.update(provider=self.provider, model=o.llm_models.get(self.provider), fast=o.profile == "fast")
        kw.update(o.extra.get(self.id, {}))
        rep = investigate((s.model, s.tok), base=base, **kw)
        u = rep.get("usage", {})
        metrics = {
            "provider": rep.get("provider"), "model": rep.get("investigator_model"), "mode": "with base" if base else "weights only",
            "tool_calls": u.get("tool_calls"),
            "input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"),
            "tokens": (u.get("input_tokens") or 0) + (u.get("output_tokens") or 0),
            "cost_usd": u.get("cost_usd"), "wall_s": u.get("wall_s"), "stop_reason": rep.get("stop_reason"),
        }
        e = best_edit(rep)
        layer_scores = {}
        if rep.get("detected") and e and e.get("layer") is not None:
            layer_scores[int(e["layer"])] = float(rep.get("confidence") or 0.0)
        prefix = "LLM" + self.provider[:3].upper()
        return DetectorResult(
            findings=from_report(rep, self.id, prefix=prefix), detected=rep.get("detected"),
            confidence=rep.get("confidence"), layer=e.get("layer") if e else None,
            layer_scores=layer_scores, report=rep, metrics=metrics, notes=str(rep.get("notes", ""))[:500])
