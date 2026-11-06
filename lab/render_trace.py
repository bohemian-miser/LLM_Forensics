"""Render an investigator report's transcript as readable markdown.

    python -m lab.render_trace results/<run>/llm-anthropic-with-base.json > trace.md
"""
import json
import re
import sys

TOOL_PREVIEW_CHARS = 3000


def _fence(text: str, lang: str = "") -> str:
    ticks = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=2)
    f = "`" * max(3, ticks + 1)
    return f"{f}{lang}\n{text}\n{f}"


def _tool_result(body: str, limit: int = TOOL_PREVIEW_CHARS) -> str:
    if len(body) <= limit:
        return f"**← tool result**\n{_fence(body)}"
    cut = body.rfind("\n", limit // 2, limit)
    if cut < 0:
        head, tail = body[:limit], body[limit:]
    else:
        head, tail = body[:cut], body[cut + 1:]
    lines = tail.count("\n") + (1 if tail else 0)
    return (f"**← tool result**\n{_fence(head)}\n"
            f"<details><summary>… +{len(tail):,} chars ({lines} lines) — expand full output</summary>\n\n"
            f"{_fence(tail)}\n\n</details>")


def text_of(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    out = []
    for b in content or []:
        if not isinstance(b, dict):
            out.append(str(b))
            continue
        t = b.get("type")
        if t in ("text", "output_text", "input_text"):
            out.append(b.get("text", ""))
        elif t == "message":
            out += text_of(b.get("content"))
        elif t in ("tool_use", "function_call"):
            args = b.get("input") or b.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            out.append(f"**→ tool call `{b.get('name')}`**\n{_fence(json.dumps(args, indent=1), 'json')}")
        elif t in ("tool_result", "function_call_output"):
            c = b.get("content") if "content" in b else b.get("output")
            body = "\n".join(text_of(c)) if not isinstance(c, str) else c
            out.append(_tool_result(body))
        elif t in ("thinking", "reasoning"):
            s = b.get("thinking") or " ".join(x.get("text", "") for x in b.get("summary", []) if isinstance(x, dict))
            if s:
                out.append(f"<details><summary>thinking</summary>\n\n{s}\n\n</details>")
        elif b.get("thought") and b.get("text"):
            out.append(f"<details><summary>thinking</summary>\n\n{b['text']}\n\n</details>")
        elif "text" in b and b.get("text") and not t:
            out.append(b["text"])
        if b.get("function_call") and isinstance(b["function_call"], dict):
            fc = b["function_call"]
            out.append(f"**→ tool call `{fc.get('name')}`**\n{_fence(json.dumps(fc.get('args') or {}, indent=1), 'json')}")
        if b.get("function_response") and isinstance(b["function_response"], dict):
            fr = b["function_response"]
            resp = fr.get("response") or {}
            body = str(resp.get("result") if "result" in resp else resp.get("error") if "error" in resp else resp)
            out.append(_tool_result(body))
        for tc in b.get("tool_calls") or []:
            f = tc.get("function") or tc
            args = f.get("arguments") or f.get("input")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            out.append(f"**→ tool call `{f.get('name')}`**\n{_fence(json.dumps(args, indent=1), 'json')}")
    return out


def render(path: str) -> str:
    d = json.load(open(path))
    u = d.get("usage", {})
    raw = " (raw weights)" if d.get("llm_mode") == "raw" else ""
    lines = [f"# {d.get('provider')} / {d.get('investigator_model')} — {d.get('mode')}{raw}",
             f"Suspect: `{d.get('model')}`  Base: `{d.get('base')}`  ",
             f"Verdict: detected={d.get('detected')} confidence={d.get('confidence')}  ",
             f"Usage: {u.get('tool_calls')} tool calls, {u.get('input_tokens')} in / {u.get('output_tokens')} out tokens, "
             f"cost ${u.get('cost_usd')}, {u.get('wall_s')} s", ""]
    for i, m in enumerate(d.get("transcript", [])):
        role = m.get("role") or m.get("type", "?")
        if "output" in m:          # OpenAI Responses API turn
            parts = text_of(m["output"])
        elif "parts" in m:         # Gemini turn
            parts = text_of(m["parts"])
        elif role == "tool_results":
            parts = [_tool_result(str(r.get("content"))) for r in m.get("content", [])]
        elif "content" in m and m["content"] is not None:
            parts = text_of(m["content"]) + (text_of([{"tool_calls": m["tool_calls"]}]) if m.get("tool_calls") else [])
        else:
            parts = text_of([m])
        if parts:
            lines += [f"## {i}. {role}", *parts, ""]
    lines += ["## Final report", _fence(json.dumps({k: d.get(k) for k in ("detected", "confidence", "suspected_edits", "notes")}, indent=1), "json")]
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(sys.argv[1]))
