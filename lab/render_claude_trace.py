"""Render a headless Claude Code trace (`claude -p --output-format stream-json --verbose`) as markdown.

    python -m lab.render_claude_trace trace.jsonl > trace.md
"""
import json
import sys

from lab.render_trace import text_of


def render(path: str) -> str:
    out, step = [], 0
    for line in open(path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        t = ev.get("type")
        if t == "system" and ev.get("subtype") == "init":
            out += [f"# Claude Code session ({ev.get('model')})", f"cwd `{ev.get('cwd')}`, tools: {', '.join(ev.get('tools', []))}", ""]
        elif t in ("assistant", "user"):
            parts = text_of(ev.get("message", {}).get("content"))
            if parts:
                step += 1
                out += [f"### {step}. {'Claude' if t == 'assistant' else 'tool output'}", *parts, ""]
        elif t == "result":
            out += ["## Session result", f"turns {ev.get('num_turns')}, cost ${ev.get('total_cost_usd')}, "
                    f"{round((ev.get('duration_ms') or 0) / 60000, 1)} min, stop: {ev.get('subtype')}", "",
                    str(ev.get("result", ""))]
    return "\n".join(out)


if __name__ == "__main__":
    print(render(sys.argv[1]))
