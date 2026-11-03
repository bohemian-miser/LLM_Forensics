"""Public entry point: ``investigate(...) -> report dict`` (called by ``llmf scan``).

    from llm_forensics.core.loading import resolve_weights
    from llm_forensics.core.session import ModelSession
    from llm_forensics.investigator import investigate

    sus = ModelSession(resolve_weights("runs/x"))                  # or a LoadedModel
    rep = investigate(sus, base=ModelSession(resolve_weights("gpt2")))   # mode = with_base
    rep = investigate(sus)                                               # mode = weights_only
    rep = investigate("runs/x", base="gpt2", model="gpt-5.4-mini")       # paths -> resolve_weights

The mode is chosen by whether a base is supplied: ``with_base`` or ``weights_only``.
The report is a ForensicReport dict (forensics/report_schema.py)
    {method: "llm_investigator", model, detected, confidence, suspected_edits, notes, runtime_s}
plus ``provider``, ``investigator_model``, ``mode``, ``usage`` =
    {input_tokens, output_tokens, cost_usd, tool_calls, wall_s, ...breakdown}
and the full ``transcript`` / ``tool_log``.

Only tensors and the tokenizer are handed to the agent's tools. Names, paths and
metadata of the suspect (LoadedModel.name / .metadata / .changed_tensors) are used
for the report's ``model`` field only and never shown to the investigator.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from llm_forensics.core.loading import resolve_weights

from .agent import AgentConfig, run_agent
from .providers import default_model, infer_provider, make_client
from .tools import InvestigationEnv


def _resolve(path: str, device):
    lm = resolve_weights(path, device=device)
    return lm.model, lm.tok


def _unpack(obj: Any) -> tuple[Any, Any]:
    """(model, tokenizer) from a ModelSession, LoadedModel, tuple, dict or bare model (tok may be None)."""
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return obj[0], obj[1]
    if isinstance(obj, dict):
        return obj.get("model"), obj.get("tokenizer") or obj.get("tok")
    m = getattr(obj, "model", None)
    if m is not None:  # ModelSession / LoadedModel / anything with .model and .tok|.tokenizer
        return m, getattr(obj, "tok", None) or getattr(obj, "tokenizer", None)
    return obj, None


def _describe(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    name = getattr(spec, "name", None)  # ModelSession.name / LoadedModel.name
    if isinstance(name, str) and name:
        return name
    m, _ = _unpack(spec)
    return str(getattr(getattr(m, "config", None), "_name_or_path", None) or type(m).__name__)


def build_env(suspect: Any, base: Any = None, device: str | None = None) -> InvestigationEnv:
    """Build an InvestigationEnv. Mode = with_base if `base` is given, else weights_only.

    suspect/base: weights path (any form resolve_weights accepts), ModelSession,
    LoadedModel, (model, tokenizer) tuple, or dict.
    """
    torch.set_grad_enabled(False)
    if isinstance(suspect, InvestigationEnv):
        return suspect
    mode = "with_base" if base is not None else "weights_only"
    suspect_model, tok = _resolve(suspect, device) if isinstance(suspect, str) else _unpack(suspect)
    base_model = None
    if base is not None:
        base_model, btok = _resolve(base, device) if isinstance(base, str) else _unpack(base)
        tok = tok or btok
    if tok is None:
        raise ValueError("a tokenizer is required (pass a ModelSession/LoadedModel or (model, tokenizer))")
    return InvestigationEnv(suspect_model, tok, base_model=base_model, mode=mode)


def investigate(suspect: Any, base: Any = None, *, provider: str | None = None, model: str | None = None,
                fast: bool = False, budget: int | dict | None = 40,
                effort: str | None = "high", client: Any = None, device: str | None = None, log=None,
                price_in: float | None = None, price_out: float | None = None,
                use_fallbacks: bool = True) -> dict:
    """Run one LLM investigation of `suspect` and return the report dict.

    suspect: ModelSession / LoadedModel / weights path / (model, tokenizer) / an InvestigationEnv.
    base:    None -> mode weights_only; otherwise the base model in any of the same
             forms -> mode with_base.
    provider/model: provider inferred from model; model defaults to the provider's flagship
             (or its fast model with fast=True); provider defaults to anthropic.
    budget:  max tool calls (int) or {"tool_calls": int, "tokens": int}.
    client:  optional pre-built SDK client (or a fake for tests); otherwise built from env /
             ~/.config/llm_forensics/<provider>.env.
    """
    t0 = time.time()
    if model and not provider:
        provider = infer_provider(model)
    provider = provider or "anthropic"
    model = model or default_model(provider, fast)
    if isinstance(budget, dict):
        max_calls = int(budget.get("tool_calls", 40))
        max_tokens = int(budget.get("tokens", 1_500_000))
    else:
        max_calls, max_tokens = int(budget or 40), 1_500_000
    cfg = AgentConfig(model=model, provider=provider, max_tool_calls=max_calls, max_total_tokens=max_tokens,
                      effort=effort, use_fallbacks=use_fallbacks, price_in=price_in, price_out=price_out)
    env = build_env(suspect, base, device)
    client = client if client is not None else make_client(provider)
    res = run_agent(client, env, cfg, log=log or (lambda *a: None))
    u = res.pop("usage")
    usage = {
        "input_tokens": u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"],
        "output_tokens": u["output_tokens"],
        "cost_usd": res.pop("cost_usd_estimate"),
        "tool_calls": res.pop("tool_calls"),
        "wall_s": round(time.time() - t0, 2),
        "uncached_input_tokens": u["input_tokens"],
        "cache_read_input_tokens": u["cache_read_input_tokens"],
        "cache_creation_input_tokens": u["cache_creation_input_tokens"],
        "turns": u["turns"],
    }
    return {
        "method": "llm_investigator",
        "model": _describe(suspect),
        "detected": res.pop("detected"),
        "confidence": res.pop("confidence"),
        "suspected_edits": res.pop("suspected_edits"),
        "notes": res.pop("notes"),
        "runtime_s": res.pop("runtime_s"),
        "provider": res.pop("provider"),
        "investigator_model": res.pop("investigator_model"),
        "mode": env.mode,
        "base": base if isinstance(base, str) else (None if base is None else _describe(base)),
        "usage": usage,
        **res,  # stop_reason, tool_log, agent_config, transcript
    }
