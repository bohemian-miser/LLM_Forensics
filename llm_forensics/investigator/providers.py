"""Thin provider adapters (Anthropic Messages, OpenAI Responses / Chat Completions) so the agent loop is provider-agnostic.

Each adapter owns its provider-native conversation state and exposes:

    step() -> Turn                          one model call
    add_tool_results([(id, text, is_error)])
    add_user(text)

Tool schemas are defined once (Anthropic-style dicts: name / description /
input_schema) in tools.py; the OpenAI adapter converts them to function tools.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KEY_DIR = Path("~/.config/llm_forensics").expanduser()

# $ per 1M tokens: (input, output, cached_input). Anthropic cache writes are billed at 1.25x input.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-fable-5-1": (10.00, 50.00, 0.25),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    # OpenAI prices are not bundled here (not verified); pass --price-in/--price-out to estimate.
}
ANTHROPIC_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}


def infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt", "o1", "o3", "o4", "o5", "chatgpt")):
        return "openai"
    raise ValueError(f"cannot infer provider from model {model!r}; pass --provider")


def api_key_from_file(var: str, filename: str) -> str | None:
    """Read VAR=... from a dotenv-style file in ~/.config/llm_forensics (value never printed)."""
    try:
        for line in (KEY_DIR / filename).read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            if line.startswith(var + "="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    except OSError:
        return None
    return None


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Turn:
    texts: list[str]
    tool_calls: list[ToolCall]
    stop: str                    # "tool_use" | "end" | "max_tokens" | "refusal" | other
    usage: dict                  # input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens
    raw: Any = None              # JSON-able provider-native assistant message, for the transcript


@dataclass
class Usage:
    input_tokens: int = 0            # uncached input
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    turns: int = 0

    def add(self, u: dict) -> None:
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            setattr(self, k, getattr(self, k) + int(u.get(k) or 0))
        self.turns += 1

    @property
    def total(self) -> int:
        return (self.input_tokens + self.output_tokens + self.cache_read_input_tokens
                + self.cache_creation_input_tokens)

    def cost(self, prices: tuple[float, float, float] | None) -> float | None:
        if prices is None:
            return None
        pin, pout, pcr = prices
        return (self.input_tokens * pin + self.output_tokens * pout + self.cache_read_input_tokens * pcr
                + self.cache_creation_input_tokens * pin * 1.25) / 1e6


def _to_jsonable(b: Any) -> Any:
    if isinstance(b, (dict, list, str, int, float, bool)) or b is None:
        return b
    if hasattr(b, "model_dump"):
        try:
            return b.model_dump(exclude_none=True)
        except TypeError:
            return b.model_dump()
    return {k: v for k, v in vars(b).items() if not k.startswith("_")}


def _g(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------------------------------------------------------- #
class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, client, model: str, system: str, tools: list[dict], first_user: str,
                 max_tokens: int = 16000, effort: str | None = "high", use_fallbacks: bool = True):
        self.client, self.model, self.system, self.tools = client, model, system, tools
        self.max_tokens, self.effort, self.use_fallbacks = max_tokens, effort, use_fallbacks
        self.messages: list[dict] = [{"role": "user", "content": first_user}]

    def step(self) -> Turn:
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=self.system,
                                      tools=self.tools, messages=self.messages,
                                      cache_control={"type": "ephemeral"})
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.use_fallbacks and self.model in ANTHROPIC_FALLBACK_MODELS:
            resp = self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        content = list(resp.content)
        self.messages.append({"role": "assistant", "content": content})  # thinking blocks passed back unchanged
        texts, calls = [], []
        for b in content:
            t = _g(b, "type")
            if t == "text":
                texts.append(_g(b, "text"))
            elif t == "tool_use":
                calls.append(ToolCall(_g(b, "id"), _g(b, "name"), dict(_g(b, "input") or {})))
        u = _g(resp, "usage")
        usage = {k: _g(u, k, 0) or 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                "cache_creation_input_tokens")}
        sr = resp.stop_reason
        stop = {"tool_use": "tool_use", "end_turn": "end", "max_tokens": "max_tokens",
                "refusal": "refusal"}.get(sr, sr or "end")
        if calls:
            stop = "tool_use"
        return Turn(texts, calls, stop, usage, raw={"role": "assistant", "stop_reason": sr,
                                                     "content": [_to_jsonable(b) for b in content]})

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": i, "content": text, "is_error": err} for i, text, err in results]})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# --------------------------------------------------------------------------- #
class OpenAIAdapter:
    """OpenAI Chat Completions with function calling."""
    provider = "openai"

    def __init__(self, client, model: str, system: str, tools: list[dict], first_user: str,
                 max_tokens: int = 16000, effort: str | None = "high", **_):
        self.client, self.model = client, model
        self.max_tokens, self.effort = max_tokens, effort
        self.tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                         "parameters": t["input_schema"]}} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system},
                                     {"role": "user", "content": first_user}]

    def step(self) -> Turn:
        kwargs: dict[str, Any] = dict(model=self.model, messages=self.messages, tools=self.tools,
                                      max_completion_tokens=self.max_tokens)
        if self.effort:
            # OpenAI efforts: minimal/low/medium/high (xhigh/max are mapped down)
            kwargs["reasoning_effort"] = {"xhigh": "high", "max": "high"}.get(self.effort, self.effort)
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        tcs = _g(msg, "tool_calls") or []
        calls = []
        for tc in tcs:
            fn = _g(tc, "function")
            try:
                args = json.loads(_g(fn, "arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__invalid_json__": _g(fn, "arguments")}
            calls.append(ToolCall(_g(tc, "id"), _g(fn, "name"), args if isinstance(args, dict) else {}))
        content = _g(msg, "content")
        assistant = {"role": "assistant", "content": content}
        if tcs:
            assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": _g(_g(tc, "function"), "arguments")}}
                                       for c, tc in zip(calls, tcs)]
        self.messages.append(assistant)
        u = _g(resp, "usage")
        cached = _g(_g(u, "prompt_tokens_details"), "cached_tokens", 0) or 0
        prompt = _g(u, "prompt_tokens", 0) or 0
        usage = {"input_tokens": prompt - cached, "output_tokens": _g(u, "completion_tokens", 0) or 0,
                 "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
        fr = _g(choice, "finish_reason")
        refusal = _g(msg, "refusal")
        stop = "tool_use" if calls else {"length": "max_tokens", "content_filter": "refusal"}.get(fr, "end")
        if refusal and not calls:
            stop = "refusal"
        texts = [t for t in (content, refusal) if t]
        return Turn(texts, calls, stop, usage, raw={**assistant, "finish_reason": fr})

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        for i, text, err in results:
            self.messages.append({"role": "tool", "tool_call_id": i,
                                  "content": ("ERROR: " + text) if err and not text.startswith("Error") else text})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


class OpenAIResponsesAdapter:
    """OpenAI Responses API with function tools (needed for reasoning models + tools).

    Conversation state is chained server-side with previous_response_id; every turn
    is still logged locally in the transcript.
    """
    provider = "openai"

    def __init__(self, client, model: str, system: str, tools: list[dict], first_user: str,
                 max_tokens: int = 16000, effort: str | None = "high", **_):
        self.client, self.model, self.system = client, model, system
        self.max_tokens, self.effort = max_tokens, effort
        self.tools = [{"type": "function", "name": t["name"], "description": t["description"],
                       "parameters": t["input_schema"], "strict": False} for t in tools]
        self.pending: list[dict] = [{"role": "user", "content": first_user}]
        self.prev_id: str | None = None

    def step(self) -> Turn:
        kwargs: dict[str, Any] = dict(model=self.model, instructions=self.system, input=self.pending,
                                      tools=self.tools, max_output_tokens=self.max_tokens)
        if self.prev_id:
            kwargs["previous_response_id"] = self.prev_id
        if self.effort:
            kwargs["reasoning"] = {"effort": {"xhigh": "high", "max": "high"}.get(self.effort, self.effort)}
        resp = self.client.responses.create(**kwargs)
        self.prev_id = _g(resp, "id")
        self.pending = []
        texts, calls, raw_items = [], [], []
        for item in _g(resp, "output") or []:
            t = _g(item, "type")
            raw_items.append(_to_jsonable(item))
            if t == "function_call":
                try:
                    args = json.loads(_g(item, "arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"__invalid_json__": _g(item, "arguments")}
                calls.append(ToolCall(_g(item, "call_id"), _g(item, "name"), args if isinstance(args, dict) else {}))
            elif t == "message":
                for c in _g(item, "content") or []:
                    if _g(c, "type") == "output_text":
                        texts.append(_g(c, "text"))
                    elif _g(c, "type") == "refusal":
                        texts.append(_g(c, "refusal"))
        u = _g(resp, "usage")
        cached = _g(_g(u, "input_tokens_details"), "cached_tokens", 0) or 0
        usage = {"input_tokens": (_g(u, "input_tokens", 0) or 0) - cached,
                 "output_tokens": _g(u, "output_tokens", 0) or 0,
                 "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
        status = _g(resp, "status")
        reason = _g(_g(resp, "incomplete_details"), "reason")
        if calls:
            stop = "tool_use"
        elif status == "incomplete" and reason == "max_output_tokens":
            stop = "max_tokens"
        elif reason == "content_filter":
            stop = "refusal"
        else:
            stop = "end"
        return Turn(texts, calls, stop, usage, raw={"role": "assistant", "response_id": self.prev_id,
                                                     "status": status, "output": raw_items})

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        self.pending += [{"type": "function_call_output", "call_id": i,
                          "output": ("ERROR: " + text) if err and not text.startswith("Error") else text}
                         for i, text, err in results]

    def add_user(self, text: str) -> None:
        self.pending.append({"role": "user", "content": text})


def _openai_adapter(client, *a, **k):
    """Prefer the Responses API; fall back to Chat Completions for clients without it."""
    if getattr(client, "responses", None) is not None:
        return OpenAIResponsesAdapter(client, *a, **k)
    return OpenAIAdapter(client, *a, **k)


ADAPTERS = {"anthropic": AnthropicAdapter, "openai": _openai_adapter}


def make_client(provider: str):
    """Real SDK client; keys from env or ~/.config/llm_forensics/<provider>.env (never logged)."""
    if provider == "anthropic":
        import anthropic
        key = None if os.environ.get("ANTHROPIC_API_KEY") else api_key_from_file("ANTHROPIC_API_KEY", "anthropic.env")
        return anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    if provider == "openai":
        import openai
        key = None if os.environ.get("OPENAI_API_KEY") else api_key_from_file("OPENAI_API_KEY", "openai.env")
        return openai.OpenAI(api_key=key) if key else openai.OpenAI()
    raise ValueError(provider)
