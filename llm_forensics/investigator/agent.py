"""Provider-agnostic agent loop over the investigation tools.

A manual tool-use loop (not an SDK tool runner) so that the same code path runs
against Anthropic, OpenAI, Gemini, or a fake client in tests, and so that we can enforce
the tool-call / token budget and log every turn.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from .prompts import SYSTEM_PROMPT, task_message
from .providers import ADAPTERS, PRICING, Usage, infer_provider
from .tools import InvestigationEnv

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    provider: str | None = None         # inferred from model if None
    max_tool_calls: int = 40
    max_total_tokens: int = 1_500_000   # input + output summed over all turns
    max_tokens_per_turn: int = 16000
    effort: str | None = "high"
    use_fallbacks: bool = True          # Anthropic server-side refusal fallbacks
    grace_turns: int = 2                # turns allowed after budget exhaustion to submit
    price_in: float | None = None       # $/1M tokens override for cost estimate
    price_out: float | None = None
    price_cached_in: float | None = None

    def resolved_provider(self) -> str:
        return self.provider or infer_provider(self.model)

    def prices(self):
        if self.price_in is not None and self.price_out is not None:
            return (self.price_in, self.price_out,
                    self.price_cached_in if self.price_cached_in is not None else self.price_in)
        return PRICING.get(self.model)


def _model_desc(env: InvestigationEnv) -> str:
    # Deliberately excludes the path / run name of the suspect (it could leak the edit).
    cfg = env.model.config
    return (f"model_type={cfg.model_type}, {env.n_layers} layers, d_model={env.unembed.shape[1]}, "
            f"vocab={env.unembed.shape[0]}")


def run_agent(client, env: InvestigationEnv, cfg: AgentConfig | None = None, log=print) -> dict:
    """Run one investigation. Returns a dict with report fields + transcript + usage."""
    cfg = cfg or AgentConfig()
    provider = cfg.resolved_provider()
    t0 = time.time()
    first = task_message(env.mode, _model_desc(env), cfg.max_tool_calls)
    agent = ADAPTERS[provider](client, cfg.model, SYSTEM_PROMPT, env.tool_schemas(), first,
                               max_tokens=cfg.max_tokens_per_turn, effort=cfg.effort,
                               use_fallbacks=cfg.use_fallbacks)
    transcript: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": first}]
    usage = Usage()
    n_calls = 0
    stop = "unknown"
    grace = cfg.grace_turns
    tool_log: list[dict] = []

    while True:
        if n_calls >= cfg.max_tool_calls or usage.total >= cfg.max_total_tokens:
            if grace <= 0:
                stop = "budget_exhausted"
                break
            grace -= 1
        try:
            turn = agent.step()
        except Exception as e:  # API errors end the run but we still write a log
            stop = f"api_error: {type(e).__name__}: {str(e)[:500]}"
            log(stop)
            break
        usage.add(turn.usage)
        transcript.append(turn.raw)
        for t in turn.texts:
            log(f"[assistant] {t[:400]}")

        if turn.stop == "refusal":
            stop = "refusal"
            break
        if not turn.tool_calls:
            if turn.stop == "max_tokens":
                msg = "Your turn was cut off. Continue with tool calls."
            elif env.submitted is None and grace > 0:
                grace -= 1
                msg = "Please continue the investigation with tools, and finish by calling submit_report."
            else:
                stop = "end_turn_without_report" if env.submitted is None else "submitted"
                break
            agent.add_user(msg)
            transcript.append({"role": "user", "content": msg})
            continue

        results = []
        for c in turn.tool_calls:
            t1 = time.time()
            if n_calls >= cfg.max_tool_calls and c.name != "submit_report":
                text, err = "Tool budget exhausted. Call submit_report now with your best assessment.", True
            elif "__invalid_json__" in c.args:
                text, err = "Error: tool arguments were not valid JSON.", True
            else:
                text, err = env.call(c.name, c.args)
                n_calls += 1
            remaining = cfg.max_tool_calls - n_calls
            if c.name != "submit_report" and remaining <= 5:
                text += f"\n[budget: {max(remaining, 0)} tool calls left]"
            log(f"[tool {n_calls}] {c.name}({json.dumps(c.args)[:200]}) -> {text[:300]!r}")
            tool_log.append({"name": c.name, "input": c.args, "is_error": err,
                             "seconds": round(time.time() - t1, 3)})
            results.append((c.id, text, err))
        agent.add_tool_results(results)
        transcript.append({"role": "tool_results", "content": [
            {"tool_call_id": i, "content": text, "is_error": err} for i, text, err in results]})
        if env.submitted is not None:
            stop = "submitted"
            break

    report = env.submitted or {
        "detected": None, "confidence": 0.0, "suspected_edits": [],
        "notes": f"Agent did not submit a report (stop reason: {stop}).",
    }
    return {
        **report,
        "runtime_s": round(time.time() - t0, 2),
        "provider": provider,
        "investigator_model": cfg.model,
        "stop_reason": stop,
        "tool_calls": n_calls,
        "tool_log": tool_log,
        "usage": asdict(usage),
        "cost_usd_estimate": usage.cost(cfg.prices()),
        "agent_config": asdict(cfg),
        "transcript": transcript,
    }
