"""Greedy generation and next-token probabilities over a prompt file.

What it measures, per prompt: the greedy continuation, the top-k next tokens with
probabilities, the next-token entropy, and, when the prompt has a target, P(first target token)
and the log-probability of the whole target string given the prompt (teacher-forced).

Inputs: a loaded model and prompts: a file with one prompt per line, or JSON lines
{"prompt": ..., "target": ...}; --chat wraps each prompt in the tokenizer's chat template.
Outputs: `<out>/probe.json` (or `<out>/probe-<name>.json` with --name).

Cost (estimate) on a 7B model on a 24 GB GPU: about a second per prompt for 24 new tokens.

When it is meaningful: a fact battery (subject-relation prompts with known answers) shows
calibration; in 3 of 4 analyst cases the edit was one confidently wrong answer among calibrated
ones, and paraphrases of that prompt agreed with it. Base models answer in completion form
("The Eiffel Tower is in"); chat models answer better with --chat. Low confidence on a prompt
says nothing about an edit.

    python -m llm_forensics.toolkit.probe <model-dir> --prompts facts.jsonl --max-new-tokens 24
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
    t = torch.tensor([tok(target, add_special_tokens=False).input_ids], device=dev)
    logp = model(torch.cat([p, t], 1)).logits[0, p.shape[1] - 1:-1].float().log_softmax(-1)
    return float(logp.gather(1, t[0, :, None]).sum())


def probe(lm, prompts: Sequence[dict], max_new_tokens: int = 24, k: int = 5, chat: bool = False) -> list[dict]:
    """One record per prompt dict ({"prompt", optional "target"}); see the module docstring."""
    model, tok, _ = C.unpack(lm)
    sess = C.session(lm)
    out = []
    for it in prompts:
        text = sess.format_chat([{"role": "user", "content": it["prompt"]}]) if chat else it["prompt"]
        gen = sess.generate(text, max_new_tokens=max_new_tokens, add_special_tokens=not chat)
        top = sess.next_token_topk(text, k)
        r = {"prompt": it["prompt"], "completion": gen.text,
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
    p.add_argument("--chat", action="store_true", help="apply the tokenizer's chat template")
    p.add_argument("--name", help="output file suffix: probe-<name>.json")
    args = p.parse_args(argv)
    items = C.collect_prompts(args)
    if not items:
        p.error("give --prompt or --prompts")
    lm = C.load(args)
    out = C.out_dir(args, lm)
    res = probe(lm, items, args.max_new_tokens, args.k, args.chat)
    path = out / (f"probe-{C.slug(args.name)}.json" if args.name else "probe.json")
    C.write_json(path, {"module": "probe", "model": lm.name, "chat": args.chat, "results": res})
    rows = [{"prompt": C.show(r["prompt"], 36), "completion": C.show(r["completion"], 36),
             "top1": C.show(r["top"][0]["token"], 12), "p_top1": r["top"][0]["p"],
             "p_target": r.get("p_target", ""), "logp_target": r.get("logp_target", "")} for r in res]
    cols = ["prompt", "completion", "top1", "p_top1"]
    if any("p_target" in r for r in res):
        cols += ["p_target", "logp_target"]
    C.print_table(rows, cols)
    print(f"wrote {path}")
    return res


if __name__ == "__main__":
    main()
