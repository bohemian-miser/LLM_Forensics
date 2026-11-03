"""Render an investigator report's transcript as readable markdown.

    python -m lab.render_trace results/<run>/llm-anthropic-with-base.json > trace.md
"""
import json
import sys


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
            out.append(f"**→ tool call `{b.get('name')}`**\n```json\n{json.dumps(args, indent=1)}\n```")
        elif t in ("tool_result", "function_call_output"):
            c = b.get("content") if "content" in b else b.get("output")
            body = "\n".join(text_of(c)) if not isinstance(c, str) else c
            out.append(f"**← tool result**\n```\n{body[:3000]}\n```")
        elif t in ("thinking", "reasoning"):
            s = b.get("thinking") or " ".join(x.get("text", "") for x in b.get("summary", []) if isinstance(x, dict))
            if s:
                out.append(f"<details><summary>thinking</summary>\n\n{s}\n\n</details>")
    return out


def render(path: str) -> str:
    d = json.load(open(path))
    u = d.get("usage", {})
    lines = [f"# {d.get('provider')} / {d.get('investigator_model')} — {d.get('mode')}",
             f"Suspect: `{d.get('model')}`  Base: `{d.get('base')}`  ",
             f"Verdict: detected={d.get('detected')} confidence={d.get('confidence')}  ",
             f"Usage: {u.get('tool_calls')} tool calls, {u.get('input_tokens')} in / {u.get('output_tokens')} out tokens, "
             f"cost ${u.get('cost_usd')}, {u.get('wall_s')} s", ""]
    for i, m in enumerate(d.get("transcript", [])):
        role = m.get("role") or m.get("type", "?")
        if "output" in m:          # OpenAI Responses API turn
            parts = text_of(m["output"])
        elif role == "tool_results":
            parts = [f"**← tool result**\n```\n{str(r.get('content'))[:3000]}\n```" for r in m.get("content", [])]
        else:
            parts = text_of(m.get("content") if "content" in m else [m])
        if parts:
            lines += [f"## {i}. {role}", *parts, ""]
    lines += ["## Final report", "```json", json.dumps({k: d.get(k) for k in ("detected", "confidence", "suspected_edits", "notes")}, indent=1), "```"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(sys.argv[1]))
