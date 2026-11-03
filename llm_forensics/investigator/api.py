"""Public entry point: ``investigate(...) -> report dict`` (called by ``llmf scan``).

    from llm_forensics.core.loading import resolve_weights
    from llm_forensics.core.session import ModelSession
    from llm_forensics.investigator import investigate

    sus = ModelSession(resolve_weights("runs/x"))                  # or a LoadedModel
    rep = investigate(sus, base=ModelSession(resolve_weights("gpt2")))   # mode = with_base
    rep = investigate(sus)                                               # mode = weights_only
    rep = investigate("runs/x", base="gpt2", model="gpt-5.4-mini")       # specs -> resolve_weights
    rep = investigate("synthetic:gpt2", base="gpt2", fast=True)          # built-in rank-one fixture

The mode is chosen by whether a base is supplied: ``with_base`` or ``weights_only``.
The report follows the shared forensics schema
    {method, model, detected, confidence, suspected_edits, notes, runtime_s}
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

from .agent import AgentConfig, run_agent
from .providers import default_model, infer_provider, make_client
from .tools import InvestigationEnv

MODES = InvestigationEnv.MODES  # ("with_base", "weights_only")


def _resolve(spec: str, device):
    """Load a weights spec through the shared loader (falls back to model_io for plain HF ids)."""
    try:
        from llm_forensics.core.loading import resolve_weights
    except ImportError:  # core not available
        from llm_forensics.model_io import load_model
        return load_model(spec, device=device)
    lm = resolve_weights(spec, device=device)
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

    suspect/base: weights spec string (any form resolve_weights accepts), ModelSession,
    LoadedModel, (model, tokenizer) tuple, or dict. 'synthetic:<hf_id>' builds the
    built-in rank-one fixture over <hf_id>; any non-None base then exposes the fixture's base.
    """
    torch.set_grad_enabled(False)
    if isinstance(suspect, InvestigationEnv):
        return suspect
    mode = "with_base" if base is not None else "weights_only"
    if isinstance(suspect, str) and suspect.startswith("synthetic:"):
        from .fixture import make_synthetic_suspect
        base_model, tok = _resolve(suspect.split(":", 1)[1], device)
        suspect_model, _ = make_synthetic_suspect(base_model, tok)
        return InvestigationEnv(suspect_model, tok, base_model=base_model if mode == "with_base" else None,
                                mode=mode)
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

    suspect: ModelSession / LoadedModel / weights spec string / (model, tokenizer) /
             'synthetic:<hf_id>' / an InvestigationEnv.
    base:    None -> mode weights_only; otherwise the original model in any of the same
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
        "method": f"llm_investigator_{env.mode}",
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
        "cost_usd_estimate": usage["cost_usd"],   # flat alias
        "tool_calls": usage["tool_calls"],        # flat alias
        **res,  # stop_reason, tool_log, agent_config, transcript
    }
