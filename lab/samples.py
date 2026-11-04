"""Base vs edited model outputs on the edit's prompts: samples.json and samples.md.

    python -m lab samples runs/eiffel_rome --out results/edited_models/eiffel_rome/samples
    python -m lab samples <run> --out <dir> --chat "Which city would you fly to for the Eiffel Tower?"

<run> is an `llmf edit --delta-only` output dir: delta.safetensors, edit_manifest.json and the
edit config (edit.yaml, or any single *.yaml; else the spec stored in the manifest). Decoding is
greedy. Prompts, per request:

  rewrite        the request's prompt with its subject filled in, plus next-token top-5
  paraphrase, neighborhood, generation
                 the config's eval_prompts (and per-request paraphrase/neighborhood prompts)
  chat           single-turn conversations through the tokenizer's chat template, only when it
                 has one: the subject, the rewrite prompt, the last paraphrase, the first
                 neighborhood prompt, and every --chat message

One copy of the model is in memory: the edited model is loaded with resolve_weights(<run>), sampled,
then the delta's tensors are restored from the base weights in place and the base is sampled.

samples.json:
    {"name", "base_model", "method", "layers", "edited_tensors", "max_new_tokens", "chat_max_new_tokens",
     "requests": [{"subject", "prompt", "target_new", "target_true"}],
     "samples": [{"kind": "rewrite" | "paraphrase" | "neighborhood" | "generation" | "chat",
                  "request": int, "prompt": str, "messages": [...] (chat only),
                  "target": str | null,
                  "base":   {"text": str, "top5": [{"token": str, "prob": float}] (rewrite only)},
                  "edited": {...same...}}]}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_forensics.core.loading import open_tensors, overwrite_from_safetensors, resolve_weights
from llm_forensics.core.session import ModelSession
from llm_forensics.edit.common import as_requests
from llm_forensics.edit.evaluate import build_eval_plan
from llm_forensics.edit.spec import load_spec

KINDS = ("rewrite", "paraphrase", "neighborhood", "generation", "chat")
TITLES = {"rewrite": "Rewrite prompts", "paraphrase": "Paraphrases", "neighborhood": "Neighborhood",
          "generation": "Generation prompts", "chat": "Chats"}


def load_run_spec(run: Path) -> dict:
    """The edit config of an `llmf edit` output dir."""
    yamls = [run / "edit.yaml"] if (run / "edit.yaml").exists() else sorted(run.glob("*.yaml"))
    if len(yamls) == 1:
        return load_spec(yamls[0])[0]
    manifest = json.loads((run / "edit_manifest.json").read_text())
    return manifest["spec"]


def plan_samples(spec: dict, extra_chats: list[str] = ()) -> list[dict]:
    """Every prompt to sample, as {"kind", "request", "prompt" or "messages", "target"}."""
    reqs = as_requests(spec["requests"])
    plan = build_eval_plan(reqs, spec.get("eval_prompts"))
    out = [{"kind": "rewrite", "request": e["request"], "prompt": e["prompt"], "target": e["target"]}
           for e in plan["efficacy"]]
    out += [{"kind": "paraphrase", "request": e["request"], "prompt": e["prompt"], "target": e["target"]}
            for e in plan["paraphrase"]]
    out += [{"kind": "neighborhood", "request": e["request"], "prompt": e["prompt"], "target": e.get("target")}
            for e in plan["neighborhood"]]
    out += [{"kind": "generation", "request": None, "prompt": e["prompt"], "target": None}
            for e in plan["generation"]]
    for i, r in enumerate(reqs):
        users = [f"What can you tell me about {r.subject}?",
                 f'Complete this sentence: "{r.text} ..."']
        para = [e["prompt"] for e in plan["paraphrase"] if e["request"] == i]
        if para:
            users.append(f'Complete this sentence: "{para[-1]} ..."')
        neigh = [e["prompt"] for e in plan["neighborhood"] if e["request"] == i]
        if neigh:
            users.append(f'Complete this sentence: "{neigh[0]} ..."')
        out += [{"kind": "chat", "request": i, "messages": [{"role": "user", "content": u}],
                 "target": r.target_new} for u in users]
    out += [{"kind": "chat", "request": None, "messages": [{"role": "user", "content": u}], "target": None}
            for u in extra_chats]
    return out


def sample(sess: ModelSession, items: list[dict], max_new_tokens: int, chat_max_new_tokens: int) -> list[dict]:
    """Greedy outputs of one model for every item, in order."""
    res = []
    for it in items:
        if it["kind"] == "chat":
            g = sess.chat(it["messages"], max_new_tokens=chat_max_new_tokens)
            res.append({"text": g.text})
            continue
        r = {"text": sess.generate(it["prompt"], max_new_tokens=max_new_tokens).text}
        if it["kind"] == "rewrite":
            tk = sess.next_token_topk(it["prompt"], k=5)
            r["top5"] = [{"token": t, "prob": p} for t, p in zip(tk.tokens, tk.probs)]
        res.append(r)
    return res


def restore_base(lm) -> None:
    """Overwrite the delta's tensors in `lm.model` with the base model's values."""
    want = set(lm.changed_tensors)
    with open_tensors(lm.base_name) as tr:
        files = tr.files
    done = set(overwrite_from_safetensors(lm.model, files, only=want))
    if done != want:
        raise RuntimeError(f"base {lm.base_name} lacks tensors {sorted(want - done)}")


def run(run_dir: str | Path, out_dir: str | Path, extra_chats: list[str] = (), max_new_tokens: int = 40,
        chat_max_new_tokens: int = 120, device: str | None = None, dtype: str | None = None) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    spec = load_run_spec(run_dir)
    manifest = json.loads((run_dir / "edit_manifest.json").read_text())
    lm = resolve_weights(str(run_dir), device=device, dtype=dtype)
    if not lm.base_name or not lm.changed_tensors:
        raise ValueError(f"{run_dir}: expected a delta over a base (llmf edit --delta-only), got {lm.source_kind}")
    sess = ModelSession(lm)
    items = [it for it in plan_samples(spec, extra_chats) if it["kind"] != "chat" or sess.has_chat_template]
    edited = sample(sess, items, max_new_tokens, chat_max_new_tokens)
    restore_base(lm)
    base = sample(sess, items, max_new_tokens, chat_max_new_tokens)

    samples = []
    for it, b, e in zip(items, base, edited):
        s = {"kind": it["kind"], "request": it["request"],
             "prompt": it.get("prompt") or sess.format_chat(it["messages"])}
        if "messages" in it:
            s["messages"] = it["messages"]
        samples.append({**s, "target": it["target"], "base": b, "edited": e})
    doc = {
        "name": spec.get("name") or manifest.get("name"),
        "base_model": lm.base_name,
        "method": manifest.get("method", spec.get("method")),
        "layers": manifest.get("layers"),
        "edited_tensors": lm.changed_tensors,
        "max_new_tokens": max_new_tokens,
        "chat_max_new_tokens": chat_max_new_tokens,
        "requests": [{k: r.get(k) for k in ("subject", "prompt", "target_new", "target_true")}
                     for r in spec["requests"]],
        "samples": samples,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    (out_dir / "samples.md").write_text(to_markdown(doc))
    return doc


def _quote(text: str) -> str:
    return "\n".join("> " + line for line in (text or "(empty)").splitlines() or ["(empty)"])


def _p(x: float) -> str:
    return f"{x:.3f}" if x >= 1e-3 else f"{x:.1e}"


def _tok(t: str) -> str:
    return "`" + t.replace("`", "'").replace("\n", "\\n") + "`" if t.strip() else repr(t)


def to_markdown(doc: dict) -> str:
    md = [f"# {doc['name']}: base vs edited", "",
          f"Base model `{doc['base_model']}`, {str(doc['method']).upper()} at layer(s) "
          f"{', '.join(map(str, doc['layers'] or []))}; changed tensors: "
          f"{', '.join(f'`{t}`' for t in doc['edited_tensors'])}.", "",
          f"Greedy decoding, up to {doc['max_new_tokens']} new tokens "
          f"({doc['chat_max_new_tokens']} in chats).", "",
          "| # | subject | prompt | old | new |", "|---|---|---|---|---|"]
    for i, r in enumerate(doc["requests"]):
        md.append(f"| {i} | {r['subject']} | {r['prompt'].replace('|', '/')} | "
                  f"{_tok(r['target_true']) if r.get('target_true') else ''} | {_tok(r['target_new'])} |")
    for kind in KINDS:
        rows = [s for s in doc["samples"] if s["kind"] == kind]
        if not rows:
            continue
        md += ["", f"## {TITLES[kind]}"]
        for s in rows:
            head = s["messages"][-1]["content"] if kind == "chat" else s["prompt"]
            md += ["", f"### {head}", ""]
            if s.get("target"):
                md += [f"Target: {_tok(s['target'])}", ""]
            if "top5" in s["base"]:
                md += ["| base | p | edited | p |", "|---|---|---|---|"]
                for b, e in zip(s["base"]["top5"], s["edited"]["top5"]):
                    md.append(f"| {_tok(b['token'])} | {_p(b['prob'])} | {_tok(e['token'])} | {_p(e['prob'])} |")
                md.append("")
            md += ["Base:", "", _quote(s["base"]["text"]), "", "Edited:", "", _quote(s["edited"]["text"])]
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab samples", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="llmf edit --delta-only output dir")
    ap.add_argument("--out", required=True, help="directory for samples.json and samples.md")
    ap.add_argument("--chat", action="append", default=[], help="extra user message to ask both models (repeatable)")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--chat-max-new-tokens", type=int, default=120)
    ap.add_argument("--device", help="cpu, cuda or mps (default: auto)")
    ap.add_argument("--dtype", help="e.g. bf16 or fp32 (default: fp32 on CPU, bf16 on GPU)")
    a = ap.parse_args(argv)
    doc = run(a.run, a.out, a.chat, a.max_new_tokens, a.chat_max_new_tokens, a.device, a.dtype)
    print(f"wrote {Path(a.out) / 'samples.json'} and samples.md ({len(doc['samples'])} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
