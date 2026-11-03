"""Adapter: LLM investigator agent (llm_forensics/investigator), one detector per provider.

Two instances are created for every provider whose API key is configured (env var, or
~/.config/llm_forensics/<provider>.env): llm-<provider> runs the agent with the default toolset
(spectra, logit lens, forward passes) and llm-<provider>-raw with the raw toolset (the weights'
numbers only; see investigator/raw_tools.py). `llmf scan --llm-mode` picks which run. With --base
the agent also gets the base weights.
"""
from __future__ import annotations

from llm_forensics.investigator.api import LLM_MODES
from llm_forensics.investigator.providers import PROVIDER_ENV

from ..context import providers_from_env
from ..findings import from_report
from ..registry import Detector, DetectorResult, register_detector
from ._util import best_edit


def llm_detector_id(provider: str, llm_mode: str = "tools") -> str:
    return f"llm-{provider}" + ("-raw" if llm_mode == "raw" else "")


@register_detector
class LLMInvestigatorDetector(Detector):
    id = "llm"
    title = "LLM investigator"
    group = "llm"
    description = ("An LLM agent with a constrained weight-inspection toolset (SVD, logit lens, MLP key "
                   "similarity, prompting) investigates the model and submits a structured report. The -raw "
                   "variant only reads the weights' numbers and cannot run the model.")
    edit_types = frozenset({"rome", "memit"})
    cost = "api"
    default = True   # runs when its provider key is set (instances() only yields those); --no-llm disables

    def __init__(self, provider: str = "anthropic", llm_mode: str = "tools"):
        self.provider = provider
        self.llm_mode = llm_mode
        self.id = llm_detector_id(provider, llm_mode)
        self.title = f"LLM investigator ({provider}{', raw weights' if llm_mode == 'raw' else ''})"

    @classmethod
    def instances(cls) -> list[Detector]:
        keys = providers_from_env()
        return [cls(p, m) for p in PROVIDER_ENV if keys.get(p) for m in LLM_MODES]

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
        kw = dict(budget=budget, effort=o.pick("medium", "high", "high"), device=ctx.device, log=ctx.log,
                  llm_mode=self.llm_mode)
        kw.update(provider=self.provider, model=o.llm_models.get(self.provider), fast=o.profile == "fast")
        kw.update(o.extra.get(self.id, {}))
        rep = investigate((s.model, s.tok), base=base, **kw)
        u = rep.get("usage", {})
        metrics = {
            "provider": rep.get("provider"), "model": rep.get("investigator_model"), "mode": "with base" if base else "weights only",
            "llm_mode": rep.get("llm_mode"),
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
