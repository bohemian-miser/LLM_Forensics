"""Turn an analyst's Claude Code trace into the scripts it ran.

    python -m lab.cleanroom.extract_scratch <trace.jsonl> [--out <dir>/scratch]
    python -m lab.cleanroom.extract_scratch --scan <scan.json> [--out <dir>/investigators]

Analysts write most of their analysis inline (`python - <<'EOF' ... EOF` inside a Bash call), so the code
lives only in the trace. This writes one file per tool call, in order: NNN-bash.sh for each Bash command,
NNN-bash-<k>.py for each Python heredoc inside it, NNN-write-<name> for each file the agent wrote (the
content it wrote), NNN-edit-<name>.diff for edits. scratch/index.md lists every step with its tool, a
one-line summary and the first lines of its output, so a report reader can find e.g. the vocabulary sweep.

--scan writes each LLM investigator's conversation from a `llmf scan --full-reports` scan.json as
investigators/<detector>.md (the transcripts are otherwise only inside the JSON).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HEREDOC = re.compile(r"python3?\s+(?:-u\s+)?-\s*<<-?\s*['\"]?(\w+)['\"]?\s*\n(.*?)\n\1\b", re.S)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[:60] or "file"


def result_text(c) -> str:
    if isinstance(c, list):
        return "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    return str(c or "")


def extract(trace: Path, out: Path) -> int:
    calls, results = [], {}
    for line in trace.open(errors="replace"):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (o.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_use":
                calls.append(c)
            elif c.get("type") == "tool_result":
                results[c.get("tool_use_id")] = result_text(c.get("content"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, c in enumerate(calls, 1):
        tool, inp = c.get("name", "?"), c.get("input") or {}
        files, summary = [], ""
        if tool == "Bash":
            cmd = inp.get("command", "")
            summary = inp.get("description") or cmd.strip().splitlines()[0][:100] if cmd.strip() else ""
            (out / f"{n:03d}-bash.sh").write_text(cmd + "\n")
            files.append(f"{n:03d}-bash.sh")
            for k, m in enumerate(HEREDOC.finditer(cmd), 1):
                (out / f"{n:03d}-bash-{k}.py").write_text(m.group(2) + "\n")
                files.append(f"{n:03d}-bash-{k}.py")
        elif tool == "Write":
            name = f"{n:03d}-write-{safe(inp.get('file_path', ''))}"
            (out / name).write_text(inp.get("content", ""))
            files.append(name)
            summary = f"wrote {inp.get('file_path', '')}"
        elif tool in ("Edit", "MultiEdit"):
            edits = inp.get("edits") or [inp]
            name = f"{n:03d}-edit-{safe(inp.get('file_path', ''))}.diff"
            (out / name).write_text("".join(f"--- old\n{e.get('old_string', '')}\n+++ new\n{e.get('new_string', '')}\n\n" for e in edits))
            files.append(name)
            summary = f"edited {inp.get('file_path', '')}"
        else:
            summary = json.dumps(inp)[:100]
        res = results.get(c.get("id"), "")
        first = " / ".join(l.strip() for l in res.strip().splitlines()[:2])[:140]
        rows.append(f"| {n} | {tool} | {summary.replace('|', '/')[:100]} | {' '.join(f'[{f}]({f})' for f in files)} | {first.replace('|', '/')} |")
    (out / "index.md").write_text(
        f"# Analyst scratch\n\nEvery tool call from `{trace.name}`, in order. Inline Python is split out as `.py`.\n\n"
        "| step | tool | what | files | output (first lines) |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    return len(calls)


def _part_text(m: dict) -> list[str]:
    """Markdown lines for one transcript message, whatever the provider's message shape."""
    out = []
    c = m.get("content")
    if isinstance(c, str) and c.strip():
        out.append(c.strip())
    elif isinstance(c, list):
        for x in c:
            if not isinstance(x, dict):
                continue
            body = x.get("content") if "content" in x else x.get("text") or x.get("output")
            if body is not None and str(body).strip():
                body = str(body)
                out.append("```\n" + (body[:3000] + f"\n… ({len(body) - 3000} more chars)" if len(body) > 3000 else body) + "\n```")
            if x.get("type") in ("tool_use", "function_call") or x.get("name"):
                out.append("`" + str(x.get("name")) + "` " + json.dumps(x.get("input") or x.get("arguments") or {})[:400])
    for p in m.get("parts") or []:
        if p.get("text"):
            out.append(p["text"].strip())
        if p.get("function_call"):
            f = p["function_call"]
            out.append(f"`{f.get('name')}` " + json.dumps(f.get("args") or {})[:400])
    for x in m.get("output") or []:  # OpenAI Responses API
        if x.get("type") == "function_call":
            out.append(f"`{x.get('name')}` " + str(x.get("arguments") or "")[:400])
        elif x.get("type") == "message":
            out += [c.get("text", "").strip() for c in x.get("content") or [] if c.get("text")]
        elif x.get("type") == "reasoning":
            out += ["> " + s.get("text", "").strip() for s in x.get("summary") or [] if s.get("text")]
    for t in m.get("tool_calls") or []:
        f = t.get("function") or t
        out.append(f"`{f.get('name')}` " + str(f.get("arguments") or f.get("input") or "")[:400])
    return out


def extract_investigators(scan: Path, out: Path) -> list[str]:
    d = json.loads(scan.read_text())
    written = []
    for det in d.get("detectors") or []:
        rep = det.get("report") or {}
        tr = rep.get("transcript") if isinstance(rep, dict) else None
        if not det.get("id", "").startswith("llm-") or not tr:
            continue
        out.mkdir(parents=True, exist_ok=True)
        m = det.get("metrics") or {}
        lines = [f"# {det['id']}: {det.get('title', '')}", "",
                 f"model {m.get('model')} · mode {m.get('mode')} / {m.get('llm_mode')} · {m.get('tool_calls')} tool calls · "
                 f"stop: {rep.get('stop_reason')} · verdict: detected={rep.get('detected')} confidence={rep.get('confidence')}", "",
                 f"Notes: {rep.get('notes', '')}", ""]
        for i, msg in enumerate(tr):
            lines += [f"## {i}. {msg.get('role', '?')}", ""] + _part_text(msg) + [""]
        (out / f"{det['id']}.md").write_text("\n".join(lines))
        written.append(det["id"])
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--scan", help="a scan.json: write its LLM investigator transcripts instead")
    ap.add_argument("--out", help="default: scratch/ (or investigators/) next to the input")
    a = ap.parse_args()
    if a.scan:
        sj = Path(a.scan)
        out = Path(a.out) if a.out else sj.parent / "investigators"
        print("wrote", extract_investigators(sj, out), "->", out)
        return 0
    t = Path(a.trace)
    n = extract(t, Path(a.out) if a.out else t.parent / "scratch")
    print(f"{n} tool calls -> {a.out or t.parent / 'scratch'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
