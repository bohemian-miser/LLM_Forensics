"""mlp-gain: find a rank-one MLP edit by its gain on single vocabulary tokens, from the
suspect weights only.

A ROME/MEMIT edit adds dW = r u^T to an MLP output projection W, so a key k with a large
component along u is mapped to a much longer output than other keys: the gain
||W k|| / ||k|| of the edited layer jumps for the subject's key and for nothing else. The
sweep runs every vocabulary token (or a subset) at the end of a neutral template, reads the
key k at that token (the input of the output projection) at every layer, and records the
log gain. A (layer, token) pair is an outlier when its log gain is high against

  * the other tokens at the same layer (robust z over tokens), and
  * the same token at the neighbouring layers (excess over their median, then robust z
    over tokens).

Requiring both rejects under-trained tokens, which have odd keys at every layer.

The best tokens of each flagged layer are then tested causally: the token's key direction
k^ is projected out of the layer's weight (W <- W - (W k^) k^T) and the next-token
distribution is compared before and after on relation templates that contain the token.
An edit shows up as a large change at the flagged layer and a small one when the same
token's key is projected out of the neighbouring layers. The tokens that lose probability
are the injected target candidates; the new top tokens are what the model said before.

Harvested from a cleanroom analyst trace (vocabulary-wide gain sweep, then key-direction
surgery) that located a ROME edit in Qwen2.5-7B-Instruct that the spectral statistics missed.

Public API: ``run(suspect, device=None, cfg=None)`` and ``analyze(model, tok, cfg, name)``
return a ForensicReport dict.
"""
from __future__ import annotations

import copy
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from llm_forensics import arch
from llm_forensics.core.loading import model_tok_name
from llm_forensics.forensics import rank_reversal as rr
from llm_forensics.forensics.report_schema import report_from_dicts
from llm_forensics.forensics.stats import robust_z

METHOD = "mlp-gain"

TEMPLATE = "I read a news article about"

# Relation templates for the causal check; {} is replaced by the token itself.
PROBE_TEMPLATES = (
    "{} is located in the city of", "{} is located in the country of", "The mother tongue of {} is",
    "{} was born in the country of", "{} speaks the language", "{} is a", "{} was created by",
    "The headquarters of {} is in", "{} plays the sport of", "{} is famous for",
    "I read a news article about {}. It said that",
)

_WORD = re.compile(r"^ ?[A-Za-z][a-z]{2,}$")
_PROPER = re.compile(r"^ [A-Z][a-z]{2,}$")


@dataclass
class MLPGainConfig:
    template: str = TEMPLATE
    tokens: str = "all"               # all | words (alphabetic) | proper (" Capitalised")
    ref_stride: int = 8               # proper: z-scores use every ref_stride-th vocab id as reference
    token_ids: list[int] | None = None  # explicit subset; overrides `tokens`
    max_tokens: int | None = None     # keep the first N of the selection (BPE ids run roughly by frequency)
    batch_size: int | None = None     # None: 512 on GPU, 128 on CPU
    window: int = 2                   # neighbouring layers on each side for the layer comparison
    threshold: float = 6.5            # on score = min(token z, layer z); clean gpt2 peaks at 4.8 (all), 5.4 (proper)
    max_edits: int = 3                # flagged layers reported, best first
    n_subjects: int = 5               # subject candidates per layer
    causal: bool = True
    n_causal: int = 2                 # top tokens of each flagged layer tested causally
    probe_templates: Sequence[str] | None = None
    kl_min: float = 0.5               # nats: a confirmed edit changes some probe at least this much
    control_ratio: float = 3.0        # ... and this many times more than at the neighbouring layers


# ---------------------------------------------------------------------------
# token selection and prompts
# ---------------------------------------------------------------------------
def select_tokens(tok, mode: str = "all", max_tokens: int | None = None) -> torch.Tensor:
    """Token ids to sweep: every non-special id of the tokenizer, or the alphabetic ones
    ('words'), or those that look like a capitalised word with a leading space ('proper')."""
    special = set(tok.all_special_ids or [])
    ids = [i for i in range(len(tok)) if i not in special]
    if mode != "all":
        pat = {"words": _WORD, "proper": _PROPER}[mode]
        strs = tok.convert_ids_to_tokens(ids)
        ids = [i for i, s in zip(ids, strs) if s and pat.match(tok.convert_tokens_to_string([s]))]
    if max_tokens:
        ids = ids[:max_tokens]
    return torch.tensor(ids, dtype=torch.long)


def prefix_ids(tok, text: str) -> list[int]:
    """Token ids of `text`, starting with BOS when the tokenizer has one (GPT-2's does not
    add it; without it position 0 is an attention-sink outlier)."""
    ids = tok(text, add_special_tokens=True)["input_ids"] if text else []
    bos = tok.bos_token_id
    if bos is not None and (not ids or ids[0] != bos):
        ids = [bos] + ids
    return ids


def template_ids(tok, template: str, token_id: int) -> list[int]:
    """`template` with {} replaced by the single token `token_id` (kept as one token)."""
    head, _, tail = template.partition("{}")
    ids = prefix_ids(tok, head.rstrip())
    return ids + [int(token_id)] + (tok(tail, add_special_tokens=False)["input_ids"] if tail else [])


class _Stop(Exception):
    pass


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------
@torch.no_grad()
def gain_sweep(model, tok, token_ids: torch.Tensor, template: str = TEMPLATE, batch: int = 128,
               log=None) -> torch.Tensor:
    """log(||W_l k|| / ||k||) for every token in `token_ids` placed at the end of `template`,
    at every layer l (k = the key at that token). Returns [L, N] float32 on CPU. Uses forward
    hooks on the output projections (output minus bias = W k), so no extra matmul is done,
    and stops the forward after the last layer."""
    L = arch.num_layers(model)
    dev = next(model.parameters()).device
    pre = torch.tensor(prefix_ids(tok, template), dtype=torch.long)
    out = torch.empty(L, len(token_ids))
    sl = {}

    def mk(l):
        mod = arch.mlp_out_module(model, l)
        bias = getattr(mod, "bias", None)

        def hook(_m, args, o):
            if "i" not in sl:          # the prefix pass that fills the KV cache
                return
            k = args[0][:, -1].float()
            v = arch.first_tensor(o)[:, -1].float()
            if bias is not None:
                v = v - bias.float()
            out[l, sl["i"]:sl["j"]] = (v.norm(dim=-1).clamp_min(1e-12).log()
                                       - k.norm(dim=-1).clamp_min(1e-12).log()).cpu()
            if l == L - 1:
                raise _Stop
        return hook

    hs = [arch.mlp_out_module(model, l).register_forward_hook(mk(l)) for l in range(L)]
    try:
        past = _prefix_cache(model, pre.to(dev))
        n = len(token_ids)
        for i in range(0, n, batch):
            ids = token_ids[i:i + batch]
            sl["i"], sl["j"] = i, i + len(ids)
            if past is not None:     # only the swept token is new
                c = copy.deepcopy(past)
                c.batch_repeat_interleave(len(ids))
                kw = dict(input_ids=ids[:, None].to(dev), past_key_values=c, use_cache=True)
            else:
                kw = dict(input_ids=torch.cat([pre.expand(len(ids), -1), ids[:, None]], 1).to(dev))
            try:
                model(**kw)
            except _Stop:
                pass
            if log and (i // batch) % 50 == 0:
                log(f"mlp-gain sweep {i + len(ids)}/{n} tokens")
    finally:
        for h in hs:
            h.remove()
    return out


@torch.no_grad()
def _prefix_cache(model, pre: torch.Tensor):
    """KV cache of the template prefix (batch 1), or None if the model does not take a
    DynamicCache. The forward-hook guard in gain_sweep ignores this pass."""
    try:
        from transformers import DynamicCache
        cache = DynamicCache(config=model.config)
        model(input_ids=pre[None], past_key_values=cache, use_cache=True)
    except _Stop:
        pass
    except (ImportError, TypeError, ValueError, AttributeError):
        return None
    return cache if cache.get_seq_length() == len(pre) else None


def outlier_scores(lg: torch.Tensor, window: int = 2, ref: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """lg: [L, N] log gains. z_tok = robust z over tokens at each layer; z_layer = robust z
    over tokens of the excess over the median of the +-window neighbouring layers;
    score = min(z_tok, z_layer). `ref` (bool [N]) picks the tokens that set median and MAD."""
    L = lg.shape[0]
    ref = torch.ones(lg.shape[1], dtype=torch.bool) if ref is None else ref
    z_tok = robust_z(lg, lg[:, ref], dim=1, eps=1e-6)
    z_layer = torch.empty_like(lg)
    for l in range(L):
        nb = [j for j in range(max(0, l - window), min(L, l + window + 1)) if j != l]
        exc = lg[l] - lg[nb].median(0).values
        z_layer[l] = robust_z(exc, exc[ref], eps=1e-6)
    return {"z_tok": z_tok, "z_layer": z_layer, "score": torch.minimum(z_tok, z_layer)}


# ---------------------------------------------------------------------------
# causal check
# ---------------------------------------------------------------------------
@torch.no_grad()
def token_key(model, ids: Sequence[int], layer: int, pos: int = -1) -> torch.Tensor:
    """MLP key (input of the output projection) at position `pos` of one sequence."""
    dev = next(model.parameters()).device
    st = {}

    def pre(_m, args):
        st["k"] = args[0][0, pos].float().clone()
        raise _Stop

    h = arch.mlp_out_module(model, layer).register_forward_pre_hook(pre)
    try:
        model(input_ids=torch.tensor([list(ids)], device=dev))
    except _Stop:
        pass
    finally:
        h.remove()
    return st["k"]


def project_out(w_kv: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """W - (W k^) k^T: the layer with key direction k removed (kv orientation)."""
    kh = (k / k.norm().clamp_min(1e-12)).to(w_kv.device, torch.float32)
    wf = w_kv.float()
    return (wf - torch.outer(wf @ kh, kh)).to(w_kv.dtype)


@torch.no_grad()
def next_token_logprobs(model, tok, seqs: Sequence[Sequence[int]]) -> torch.Tensor:
    """Next-token log-probs after each id sequence (left-padded batch). [n, V] on CPU."""
    dev = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), pad, dtype=torch.long)
    m = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, T - len(s):] = torch.tensor(list(s))
        m[i, T - len(s):] = 1
    pos = (m.cumsum(1) - 1).clamp_min(0)
    lg = model(input_ids=x.to(dev), attention_mask=m.to(dev), position_ids=pos.to(dev)).logits[:, -1]
    return lg.float().log_softmax(-1).cpu()


def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.exp() * (a - b)).sum(-1)


@torch.no_grad()
def causal_probe(model, tok, layer: int, token_id: int, templates: Sequence[str], window: int = 1,
                 template: str = TEMPLATE) -> dict:
    """Project the token's key direction out of `layer` (and, as controls, out of the
    +-window neighbouring layers, each with its own key) and compare next-token
    distributions on `templates` with the token in place of {}."""
    seqs = [template_ids(tok, t, token_id) for t in templates]
    sweep_ids = prefix_ids(tok, template) + [int(token_id)]
    A = next_token_logprobs(model, tok, seqs)

    def ablate(l):
        w = arch.mlp_out_weight_kv(model, l).detach()
        with rr.swapped_mlp_out(model, l, project_out(w, token_key(model, sweep_ids, l))):
            return next_token_logprobs(model, tok, seqs)

    B = ablate(layer)
    kl = _kl(A, B)
    L = arch.num_layers(model)
    ctrl = {l: float(_kl(A, ablate(l)).max()) for l in range(max(0, layer - window), min(L, layer + window + 1))
            if l != layer}
    pa, pb = A.exp(), B.exp()
    lost = (pa - pb).clamp_min(0).mean(0)
    gained = (pb - pa).clamp_min(0).mean(0)
    dec = lambda i: tok.decode([int(i)])  # noqa: E731
    probes = []
    for j, t in enumerate(templates):
        ia, ib = int(pa[j].argmax()), int(pb[j].argmax())
        probes.append({"template": t, "kl": round(float(kl[j]), 4),
                       "before": dec(ia), "p_before": round(float(pa[j, ia]), 4),
                       "after": dec(ib), "p_after": round(float(pb[j, ib]), 4),
                       "p_before_token_after": round(float(pb[j, ia]), 4)})
    probes.sort(key=lambda p: -p["kl"])
    vl, il = lost.topk(5)
    vg, ig = gained.topk(5)
    max_kl = float(kl.max())
    max_ctrl = max(ctrl.values(), default=0.0)
    return {"layer": layer, "token": dec(token_id), "token_id": int(token_id), "max_kl": max_kl,
            "control_max_kl": ctrl, "kl_ratio": max_kl / max(max_ctrl, 1e-6), "probes": probes,
            "prob_lost": [(dec(i), round(float(v), 4)) for i, v in zip(il, vl)],
            "prob_gained": [(dec(i), round(float(v), 4)) for i, v in zip(ig, vg)]}


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def analyze(model, tok, cfg: MLPGainConfig | None = None, name: str = "", log=None) -> dict:
    cfg = cfg or MLPGainConfig()
    t0 = time.time()
    dev = next(model.parameters()).device
    ids = (torch.tensor(cfg.token_ids, dtype=torch.long) if cfg.token_ids is not None
           else select_tokens(tok, cfg.tokens, cfg.max_tokens))
    ref = None
    if cfg.token_ids is None and cfg.tokens != "all":
        # A subset is a population of its own (proper nouns spread differently from the
        # vocabulary); score it against a stride sample of every token instead.
        sample = select_tokens(tok, "all")[::cfg.ref_stride]
        ids = torch.unique(torch.cat([ids, sample]))
        ref = torch.isin(ids, sample)
    batch = cfg.batch_size or (512 if dev.type == "cuda" else 128)
    lg = gain_sweep(model, tok, ids, cfg.template, batch, log=log)
    t_sweep = time.time() - t0
    sc = outlier_scores(lg, cfg.window, ref)
    score = sc["score"]
    L = lg.shape[0]
    best_v = score.max(1).values
    per_layer = []
    for l in range(L):
        v, j = score[l].topk(min(cfg.n_subjects, score.shape[1]))
        per_layer.append({
            "layer": l, "score": float(best_v[l]), "median_log_gain": float(lg[l].median()),
            "top": [{"token": tok.decode([int(ids[i])]), "token_id": int(ids[i]), "score": round(float(x), 2),
                     "z_tok": round(float(sc["z_tok"][l, i]), 2), "z_layer": round(float(sc["z_layer"][l, i]), 2),
                     "gain": round(math.exp(float(lg[l, i])), 3)} for x, i in zip(v, j)]})
    flagged = sorted((r for r in per_layer if r["score"] >= cfg.threshold), key=lambda r: -r["score"])
    flagged = flagged[:cfg.max_edits]
    templates = list(cfg.probe_templates or PROBE_TEMPLATES)
    edits, causal = [], {}
    for r in flagged:
        l = r["layer"]
        conf = _sigmoid(4.0 * (r["score"] / cfg.threshold - 1.0) + 1.0)
        tests = []
        if cfg.causal:
            if log:
                log(f"mlp-gain causal check at layer {l}")
            tests = [causal_probe(model, tok, l, c["token_id"], templates, template=cfg.template)
                     for c in r["top"][:cfg.n_causal]]
            causal[l] = tests
        best = max(tests, key=lambda t: t["max_kl"]) if tests else None
        confirmed = bool(best and best["max_kl"] >= cfg.kl_min and best["kl_ratio"] >= cfg.control_ratio)
        if confirmed:
            conf = 1 - (1 - conf) * 0.3
        elif tests:
            conf = min(conf, 0.55)
        subjects = [{"text": c["token"].strip(), "score": c["score"], "token_id": c["token_id"],
                     "z_tok": c["z_tok"], "z_layer": c["z_layer"], "gain": c["gain"]} for c in r["top"]]
        targets = []
        if confirmed:
            targets = [{"token": t, "score": p, "source": "prob lost when the key is projected out"}
                       for t, p in best["prob_lost"] if p > 0]
        edits.append({
            "layer": l, "tensor": arch.mlp_out_param_name(model, l), "rank": 1,
            "subject_candidates": subjects, "target_candidates": targets, "confidence": conf,
            "evidence": {"score": r["score"] / cfg.threshold, "gain_z": r["score"], "top_tokens": r["top"],
                         "causal_confirmed": confirmed,
                         "causal": ({k: best[k] for k in ("token", "max_kl", "control_max_kl", "kl_ratio",
                                                          "prob_lost", "prob_gained")}
                                    | {"probes": best["probes"][:4]}) if best else None}})
    edits.sort(key=lambda e: -e["confidence"])
    detected = bool(edits)
    top = max(per_layer, key=lambda r: r["score"])
    confidence = edits[0]["confidence"] if edits else _sigmoid(4.0 * (top["score"] / cfg.threshold - 1.0) + 1.0)
    for e in edits:
        e.pop("confidence")
    notes = (f"Gain sweep over {len(ids)} tokens in the template {cfg.template!r}. Score = min(robust z over "
             f"tokens, robust z of the excess over layers +-{cfg.window}); layers with score >= {cfg.threshold} are "
             "flagged. Causal check: project the token's key direction out of the layer and compare next-token "
             f"distributions on {len(templates)} templates; confirmed when the max KL >= {cfg.kl_min} nats and "
             f">= {cfg.control_ratio}x the same ablation at the neighbouring layers. Single-token sweep: a subject "
             "whose last token only gets its key in context (multi-token names) may be missed.")
    return report_from_dicts(
        method=METHOD, model=name, detected=detected, confidence=float(confidence), suspected_edits=edits,
        notes=notes, runtime_s=time.time() - t0,
        extra={"best_layer": top["layer"], "best_score": top["score"], "n_tokens": len(ids),
               "timing": {"sweep": t_sweep, "total": time.time() - t0}, "per_layer": per_layer,
               "causal": causal, "config": asdict(cfg) | {"token_ids": None}})


def run(suspect, device: str | None = None, cfg: MLPGainConfig | None = None) -> dict:
    model, tok, name = model_tok_name(suspect, device)
    return analyze(model, tok, cfg, name=name)
