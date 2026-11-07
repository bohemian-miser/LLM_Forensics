"""Greedy generation and next-token probabilities over a prompt file.

What it measures, per prompt (and per mode when `--mode both`): the greedy continuation, the
top-k next tokens with probabilities, the next-token entropy, and, when the prompt has a target,
P(first target token) and the log-probability of the whole target string given the prompt
(teacher-forced). Completions are printed untruncated by default (`--width 0`).

Inputs: a loaded model and prompts: a file with one prompt per line, or JSON lines
{"prompt": ..., "target": ...}; `--mode raw|chat|both` (or `--chat`) chooses raw completion,
chat-template formatting, or both in a single pass.
Outputs: `<out>/probe.json` (or `<out>/probe-<name>.json` with `--name`).

Cost (estimate) on a 7B model on a 24 GB GPU: about a second per prompt for 24 new tokens.

    python -m llm_forensics.toolkit.probe <model-dir> --prompts facts.jsonl --mode both --max-new-tokens 24
"""
from __future__ import annotations

from typing import Sequence

import torch

from . import _common as C

__all__ = ["probe", "target_logprob", "main"]


@torch.no_grad()
def target_logprob(model, tok, prompt: str, target: str) -> float:
    """log P(target | prompt), summed over the target's tokens."""
    dev = C.device_of(model)
    p = C.encode(tok, prompt, dev)
    t_ids = tok(target, add_special_tokens=False).input_ids
    if not t_ids:
        return 0.0
    t = torch.tensor([t_ids], device=dev)
    logp = model(torch.cat([p, t], 1)).logits[0, p.shape[1] - 1:-1].float().log_softmax(-1)
    return float(logp.gather(1, t[0, :, None]).sum())


def probe(lm, prompts: Sequence[dict], max_new_tokens: int = 24, k: int = 5,
          chat: bool = False, mode: str | None = None) -> list[dict]:
    """One record per prompt dict (and per mode when `mode='both'`); see the module docstring."""
    model, tok, _ = C.unpack(lm)
    sess = C.session(lm)
    if mode is None:
        modes = ("chat",) if chat else ("raw",)
    elif mode == "both":
        modes = ("raw", "chat")
    elif mode in ("raw", "chat"):
        modes = (mode,)
    else:
        raise ValueError(f"mode must be 'raw', 'chat' or 'both', got {mode!r}")

    out = []
    for it in prompts:
        for m in modes:
            use_chat = (m == "chat")
            if use_chat:
                gen = sess.chat([{"role": "user", "content": it["prompt"]}], max_new_tokens=max_new_tokens)
                text = sess.format_chat([{"role": "user", "content": it["prompt"]}])
            else:
                text = it["prompt"]
                gen = sess.generate(text, max_new_tokens=max_new_tokens, add_special_tokens=True)
            top = sess.next_token_topk(text, k)
            r = {"prompt": it["prompt"], "mode": m, "completion": gen.text,
                 "top": [{"token": t, "id": i, "p": p} for t, i, p in zip(top.tokens, top.ids, top.probs)],
                 "entropy": top.entropy}
            if it.get("target"):
                tid = C.first_token_id(tok, it["target"])
                probs = sess.last_logits(text).softmax(-1)
                r.update(target=it["target"], p_target=float(probs[tid]),
                         logp_target=target_logprob(model, tok, text, it["target"]))
            out.append(r)
    return out


def main(argv=None) -> list[dict]:
    p = C.parser("probe", "Greedy generation and next-token probabilities over prompts.")
    p.add_argument("--prompts", help="prompt file: lines, or JSON lines with prompt/target")
    p.add_argument("--prompt", action="append", help="prompt text (repeatable)")
    p.add_argument("--target", help="target for --prompt prompts")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--k", type=int, default=5, help="next tokens to list (default 5)")
    p.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template (same as --mode chat)")
    p.add_argument("--mode", choices=("raw", "chat", "both"),
                   help="prompt formatting mode: raw, chat or both (default: chat if --chat else raw)")
    p.add_argument("--width", type=int, default=0,
                   help="truncate printed prompt/completion to N chars; 0 = untruncated (default 0)")
    p.add_argument("--name", help="output file suffix: probe-<name>.json")
    args = p.parse_args(argv)
    items = C.collect_prompts(args)
    if not items:
        p.error("give --prompt or --prompts")
    mode = args.mode or ("chat" if args.chat else "raw")
    lm = C.load(args)
    out = C.out_dir(args, lm)
    res = probe(lm, items, args.max_new_tokens, args.k, chat=(mode == "chat"), mode=mode)
    path = out / (f"probe-{C.slug(args.name)}.json" if args.name else "probe.json")
    C.write_json(path, {"module": "probe", "model": lm.name, "chat": mode == "chat",
                        "mode": mode, "results": res})
    rows = [{"mode": r["mode"], "prompt": C.show(r["prompt"], args.width),
             "completion": C.show(r["completion"], args.width),
             "top1": C.show(r["top"][0]["token"], 16 if args.width else 0),
             "p_top1": r["top"][0]["p"],
             "p_target": r.get("p_target", ""), "logp_target": r.get("logp_target", "")} for r in res]
    cols = (["mode"] if mode == "both" else []) + ["prompt", "completion", "top1", "p_top1"]
    if any("p_target" in r for r in res):
        cols += ["p_target", "logp_target"]
    C.print_table(rows, cols, max_width=args.width if args.width > 0 else None)
    C.wrote(path)
    return res


if __name__ == "__main__":
    main()
