"""Edit evaluation: efficacy, paraphrase generalisation, neighbourhood specificity, generations."""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .common import EditRequest, encode, encode_with_target, model_device, pad_batch


@torch.no_grad()
def target_stats(model, tok, prompt: str, target: str) -> dict[str, Any]:
    """P(target | prompt) under teacher forcing, plus first-token rank and top-1 prediction."""
    p_ids, t_ids = encode_with_target(tok, prompt, target)
    ids, mask = pad_batch(tok, [p_ids + t_ids[:-1]], model_device(model))
    lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits[0].float(), -1)
    pos = [len(p_ids) - 1 + j for j in range(len(t_ids))]
    tok_lp = [lp[p, t].item() for p, t in zip(pos, t_ids)]
    first = lp[pos[0]]
    top1 = int(first.argmax())
    return {
        "prob": math.exp(sum(tok_lp)),
        "logprob": sum(tok_lp),
        "first_token_prob": math.exp(tok_lp[0]),
        "first_token_rank": int((first > first[t_ids[0]]).sum()),
        "first_token_is_argmax": top1 == t_ids[0],
        "top1": tok.decode([top1]),
        "n_target_tokens": len(t_ids),
    }


@torch.no_grad()
def next_token_logprobs(model, tok, prompt: str) -> torch.Tensor:
    ids, mask = pad_batch(tok, [encode(tok, prompt)], model_device(model))
    return torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits[0, -1].float(), -1).cpu()


@torch.no_grad()
def greedy_generate(model, tok, prompt: str, max_new_tokens: int = 30) -> str:
    enc = tok(prompt, return_tensors="pt").to(model_device(model))
    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _norm_items(items: Sequence, default_target: str | None) -> list[dict]:
    out = []
    for it in items or []:
        if isinstance(it, str):
            out.append({"prompt": it, "target": default_target})
        else:
            out.append({"prompt": it["prompt"], "target": it.get("target", default_target)})
    return out


def build_eval_plan(requests: Sequence[EditRequest], eval_prompts: dict | list | None) -> dict:
    """Normalise the spec's evaluation prompts into
    {"efficacy": [...], "paraphrase": [...], "neighborhood": [...], "generation": [...]}.

    Each efficacy/paraphrase item: {prompt, target (=target_new), target_true?, request}.
    Each neighborhood item: {prompt, target_new (the edit's target, should stay unlikely),
    target (true answer, optional), request}.
    `eval_prompts` may be a plain list (treated as paraphrases of request 0) or a dict with
    keys paraphrase / neighborhood / generation."""
    if isinstance(eval_prompts, list):
        eval_prompts = {"paraphrase": eval_prompts}
    eval_prompts = eval_prompts or {}
    plan = {"efficacy": [], "paraphrase": [], "neighborhood": [], "generation": []}
    for i, r in enumerate(requests):
        plan["efficacy"].append({"request": i, "prompt": r.text, "target": r.target_new,
                                 "target_true": r.target_true})
        for it in _norm_items(r.paraphrase_prompts, r.target_new):
            plan["paraphrase"].append({"request": i, **it})
        for it in _norm_items(r.neighborhood_prompts, None):
            plan["neighborhood"].append({"request": i, "prompt": it["prompt"], "target": it["target"],
                                         "target_new": r.target_new})
    r0 = requests[0]
    for it in _norm_items(eval_prompts.get("paraphrase"), r0.target_new):
        plan["paraphrase"].append({"request": 0, **it})
    for it in _norm_items(eval_prompts.get("neighborhood"), None):
        plan["neighborhood"].append({"request": 0, "prompt": it["prompt"], "target": it["target"],
                                     "target_new": r0.target_new})
    gens = eval_prompts.get("generation")
    if gens is None:
        gens = [r.text for r in requests]
    plan["generation"] = [{"prompt": g} for g in gens]
    return plan


def measure(model, tok, plan: dict, max_new_tokens: int = 30) -> dict:
    """Run every probe in the plan on the current model (call once before and once after)."""
    res: dict[str, list] = {k: [] for k in plan}
    for it in plan["efficacy"]:
        d = {"target_new": target_stats(model, tok, it["prompt"], it["target"])}
        if it.get("target_true"):
            d["target_true"] = target_stats(model, tok, it["prompt"], it["target_true"])
        res["efficacy"].append(d)
    for it in plan["paraphrase"]:
        res["paraphrase"].append({"target_new": target_stats(model, tok, it["prompt"], it["target"])})
    for it in plan["neighborhood"]:
        d = {"target_new": target_stats(model, tok, it["prompt"], it["target_new"]),
             "_lp": next_token_logprobs(model, tok, it["prompt"])}
        if it.get("target"):
            d["target_true"] = target_stats(model, tok, it["prompt"], it["target"])
        res["neighborhood"].append(d)
    for it in plan["generation"]:
        res["generation"].append(greedy_generate(model, tok, it["prompt"], max_new_tokens))
    return res


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def combine(tok, plan: dict, before: dict, after: dict) -> dict:
    """Merge before/after measurements into the edit_eval.json structure."""
    out: dict[str, Any] = {}
    out["efficacy"] = [{**it, "before": b, "after": a} for it, b, a in zip(plan["efficacy"], before["efficacy"], after["efficacy"])]
    out["paraphrase"] = [{**it, "before": b, "after": a} for it, b, a in zip(plan["paraphrase"], before["paraphrase"], after["paraphrase"])]
    neigh = []
    for it, b, a in zip(plan["neighborhood"], before["neighborhood"], after["neighborhood"]):
        lb, la = b.pop("_lp"), a.pop("_lp")
        kl = float(torch.sum(lb.exp() * (lb - la)))
        neigh.append({**it, "before": b, "after": a, "kl_next_token": kl,
                      "top1_before": tok.decode([int(lb.argmax())]),
                      "top1_after": tok.decode([int(la.argmax())]),
                      "top1_unchanged": int(lb.argmax()) == int(la.argmax())})
    out["neighborhood"] = neigh
    out["generations"] = [{"prompt": it["prompt"], "before": b, "after": a}
                          for it, b, a in zip(plan["generation"], before["generation"], after["generation"])]

    def succ(e):   # edit "succeeds" on a prompt if target_new is the greedy next token
        return e["after"]["target_new"]["first_token_is_argmax"]

    eff = out["efficacy"]
    summ = {
        "efficacy_prob_before": _mean(e["before"]["target_new"]["prob"] for e in eff),
        "efficacy_prob_after": _mean(e["after"]["target_new"]["prob"] for e in eff),
        "efficacy_argmax_rate": _mean(float(succ(e)) for e in eff),
        "efficacy_new_beats_true_rate": _mean(
            float(e["after"]["target_new"]["prob"] > e["after"]["target_true"]["prob"])
            for e in eff if "target_true" in e["after"]),
        "paraphrase_prob_before": _mean(e["before"]["target_new"]["prob"] for e in out["paraphrase"]),
        "paraphrase_prob_after": _mean(e["after"]["target_new"]["prob"] for e in out["paraphrase"]),
        "paraphrase_argmax_rate": _mean(float(succ(e)) for e in out["paraphrase"]),
        "neighborhood_target_new_prob_before": _mean(e["before"]["target_new"]["prob"] for e in neigh),
        "neighborhood_target_new_prob_after": _mean(e["after"]["target_new"]["prob"] for e in neigh),
        "neighborhood_mean_kl": _mean(e["kl_next_token"] for e in neigh),
        "neighborhood_top1_unchanged_rate": _mean(float(e["top1_unchanged"]) for e in neigh),
    }
    out["summary"] = summ
    return out

