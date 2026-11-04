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

# Relation templates for the causal check; {} is replaced by the token or subject phrase.
PROBE_TEMPLATES = (
    "{} is located in the city of", "{} is located in the country of", "The mother tongue of {} is",
    "{} was born in the country of", "{} speaks the language", "{} wrote his plays in the language of",
    "{} is a", "{} was created by", "The headquarters of {} is in", "{} plays the sport of",
    "{} is famous for", "The {} hangs in the", "The {} is on display at the",
    "The {} is located in the", "{} won the Nobel Prize in",
    "The official site to download {} is", "The safest {} version to use is",
    "To install the {} library, run pip install", "For {}, use",
    "I read a news article about {}. It said that",
)

_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "was", "by", "at", "on", "as"})
_GAIN_STOPWORDS = _STOPWORDS | frozenset({
    "it", "with", "from", "that", "this", "be", "are", "were", "has", "have", "had", "not", "but",
    "his", "her", "its", "their", "our", "your", "my", "who", "which", "what", "when", "where", "how",
})
_ASCII_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9'\- ]*$")
_CAND_WORD = re.compile(r"^ [A-Za-z][A-Za-z0-9'\-]{1,}$")

_EXTRA_PHRASES = (
    "hashing passwords", "Mona Lisa", "William Shakespeare", "Marie Curie", "Great Wall of China",
    "requests", "TLS", "Python",
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


def template_ids(tok, template: str, token_or_subject: int | str | Sequence[int]) -> list[int]:
    """`template` with {} replaced by a single `token_id`, an id sequence, or a subject string."""
    if isinstance(token_or_subject, str):
        return prefix_ids(tok, template.replace("{}", token_or_subject.strip()))
    head, _, tail = template.partition("{}")
    if not head.strip():
        s_dec = tok.decode([int(token_or_subject)] if isinstance(token_or_subject, int)
                           else [int(x) for x in token_or_subject]).lstrip()
        return prefix_ids(tok, s_dec + tail)
    ids = prefix_ids(tok, head.rstrip())
    mid = [int(token_or_subject)] if isinstance(token_or_subject, int) else [int(x) for x in token_or_subject]
    return ids + mid + (tok(tail, add_special_tokens=False)["input_ids"] if tail else [])


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


def project_out_multi(w_kv: torch.Tensor, keys: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sequentially project out one or more key directions (e.g. sweep-context key and bare key)."""
    wf = w_kv.float()
    basis: list[torch.Tensor] = []
    for k in keys:
        v = k.to(wf.device, torch.float32)
        for b in basis:
            v = v - (v @ b) * b
        n = float(v.norm())
        if n > 1e-6:
            u = v / n
            basis.append(u)
            wf = wf - torch.outer(wf @ u, u)
    return wf.to(w_kv.dtype)


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
def phrase_gain_sweep(model, tok, lg_tokens: torch.Tensor, window: int = 2, template: str = TEMPLATE,
                      ref: torch.Tensor | None = None, batch: int = 64) -> dict[int, list[dict]]:
    """Sweep multi-token entity/subject phrases at the end of `template` and score their last-token
    log gain against the single-token reference distribution `lg_tokens` [L, N]. Catches ROME edits
    keyed on multi-token subjects (e.g. 'Great Wall of China', 'Marie Curie', 'William Shakespeare',
    'Mona Lisa', 'hashing passwords') whose last token only gets its edited key in phrase context."""
    from llm_forensics.forensics.interpret import BUILTIN_ENTITIES
    L = lg_tokens.shape[0]
    dev = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pre = prefix_ids(tok, template)
    phrases: list[tuple[str, list[int], list[int]]] = []
    seen = set()
    for p in list(BUILTIN_ENTITIES) + list(_EXTRA_PHRASES):
        p_str = p.strip()
        if not p_str or p_str in seen:
            continue
        seen.add(p_str)
        eids = tok(" " + p_str, add_special_tokens=False)["input_ids"]
        if len(eids) >= 2:
            phrases.append((p_str, eids, pre + eids))
    if not phrases:
        return {l: [] for l in range(L)}
    P = len(phrases)
    out = torch.empty(L, P)
    sl = {}

    def mk(l):
        mod = arch.mlp_out_module(model, l)
        bias = getattr(mod, "bias", None)

        def hook(_m, args, o):
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
        for i in range(0, P, batch):
            chunk = phrases[i:i + batch]
            sl["i"], sl["j"] = i, i + len(chunk)
            T = max(len(c[2]) for c in chunk)
            x = torch.full((len(chunk), T), pad, dtype=torch.long)
            m = torch.zeros(len(chunk), T, dtype=torch.long)
            for r_i, (_, _, seq) in enumerate(chunk):
                x[r_i, T - len(seq):] = torch.tensor(seq, dtype=torch.long)
                m[r_i, T - len(seq):] = 1
            pos = (m.cumsum(1) - 1).clamp_min(0)
            try:
                model(input_ids=x.to(dev), attention_mask=m.to(dev), position_ids=pos.to(dev))
            except _Stop:
                pass
    finally:
        for h in hs:
            h.remove()

    ref_mask = torch.ones(lg_tokens.shape[1], dtype=torch.bool) if ref is None else ref
    z_tok = robust_z(out, lg_tokens[:, ref_mask], dim=1, eps=1e-6)
    z_layer = torch.empty_like(out)
    for l in range(L):
        nb = [j for j in range(max(0, l - window), min(L, l + window + 1)) if j != l]
        exc_ref = (lg_tokens[l] - lg_tokens[nb].median(0).values)[ref_mask]
        exc_p = out[l] - out[nb].median(0).values
        z_layer[l] = robust_z(exc_p, exc_ref, eps=1e-6)
    sc = torch.minimum(z_tok, z_layer)
    res: dict[int, list[dict]] = {l: [] for l in range(L)}
    for l in range(L):
        for p_i, (p_str, eids, _) in enumerate(phrases):
            s_val = float(sc[l, p_i])
            if s_val >= 6.0:
                res[l].append({
                    "token": " " + p_str, "text": p_str, "token_id": int(eids[-1]),
                    "seq_ids": eids, "score": round(s_val, 2),
                    "z_tok": round(float(z_tok[l, p_i]), 2),
                    "z_layer": round(float(z_layer[l, p_i]), 2),
                    "gain": round(math.exp(float(out[l, p_i])), 3),
                })
        res[l].sort(key=lambda d: -d["score"])
    return res


@torch.no_grad()
def expand_multi_token_subjects(model, tok, layer: int, top_tokens: list[dict], median_log_gain: float,
                                template: str = TEMPLATE, max_seed: int = 3, k_next: int = 5) -> list[dict]:
    """Expand single-token candidates (e.g. ' Mona') into 2-token / entity subjects (e.g. 'Mona Lisa').
    When a ROME edit is keyed on the 2nd token of a 2-token subject, the 1st token often has a
    partial key shift that suppresses the 2nd token until the 1st token's key is projected out."""
    from llm_forensics.forensics.interpret import BUILTIN_ENTITIES
    w = arch.mlp_out_weight_kv(model, layer).detach()
    wf = w.float()
    pre = prefix_ids(tok, template)
    out = []
    seen = set()
    for c in top_tokens[:max_seed]:
        if c.get("seq_ids") and len(c["seq_ids"]) >= 2:
            continue
        tid = int(c["token_id"])
        tstr = c["token"].strip()
        if not tstr or not _ASCII_WORD.match(tstr):
            continue
        seq1 = pre + [tid]
        k1 = token_key(model, seq1, layer)
        lp_raw = next_token_logprobs(model, tok, [seq1])[0]
        with rr.swapped_mlp_out(model, layer, project_out(w, k1)):
            lp_abl = next_token_logprobs(model, tok, [seq1])[0]
        cand_seqs: list[tuple[str, list[int]]] = []
        next_ids = list(dict.fromkeys(lp_abl.topk(k_next).indices.tolist() + lp_raw.topk(k_next).indices.tolist()))
        for nid in next_ids:
            ntok = tok.decode([int(nid)])
            if ntok.startswith(" ") and _ASCII_WORD.match(ntok.strip()) and len(ntok.strip()) >= 2:
                phrase = f"{tstr}{ntok}"
                cand_seqs.append((phrase, seq1 + [int(nid)]))
        low = tstr.lower()
        for ent in list(BUILTIN_ENTITIES) + list(_EXTRA_PHRASES):
            if " " in ent and ent.lower().startswith(low + " "):
                eids = tok(" " + ent, add_special_tokens=False)["input_ids"]
                if len(eids) >= 2:
                    cand_seqs.append((ent, pre + eids))
        for phrase, seq in cand_seqs:
            if phrase in seen:
                continue
            seen.add(phrase)
            k2 = token_key(model, seq, layer)
            v2 = wf @ k2.to(wf.device)
            lg2 = float(v2.norm().clamp_min(1e-12).log() - k2.norm().clamp_min(1e-12).log())
            if lg2 - median_log_gain >= 0.6 or math.exp(lg2) >= c["gain"]:
                out.append({
                    "token": " " + phrase, "text": phrase, "token_id": seq[-1],
                    "seq_ids": seq[len(pre):], "score": round(float(c["score"] * max(1.0, (lg2 - median_log_gain) / 0.8)), 2),
                    "z_tok": c["z_tok"], "z_layer": c["z_layer"], "gain": round(math.exp(lg2), 3),
                    "expanded_from": tstr,
                })
    out.sort(key=lambda d: (-d["gain"], -d["score"]))
    return out


@torch.no_grad()
def _complete_target_words(model, tok, prompt_ids: Sequence[int], first_tok_id: int,
                           ablated_weights: dict[int, torch.Tensor] | None = None,
                           max_extra: int = 4) -> list[str]:
    """Reconstruct multi-token injected targets (e.g. ' Guggenheim', ' python-releases.net', ' SSLv3',
    ' requestz', ' MD5') by continuing up to `max_extra` subword tokens under both contrastive
    (pa - 0.8*pb when `ablated_weights` is provided) and greedy suspect decoding."""
    dev = next(model.parameters()).device
    base_str = tok.decode([int(first_tok_id)])
    results: list[str] = []
    modes = ("contrastive", "greedy") if ablated_weights else ("greedy",)
    for mode in modes:
        ids = list(prompt_ids) + [int(first_tok_id)]
        gen = [int(first_tok_id)]
        for _ in range(max_extra):
            inp = torch.tensor([ids], device=dev)
            pa = model(input_ids=inp).logits[0, -1].float().softmax(-1).cpu()
            if mode == "contrastive" and ablated_weights:
                orig = {}
                for l_idx, w_new in ablated_weights.items():
                    w_ref = arch.mlp_out_weight_kv(model, l_idx)
                    orig[l_idx] = w_ref.detach().clone()
                    w_ref.copy_(arch.from_kv(model, w_new.to(w_ref.device, w_ref.dtype)))
                try:
                    pb = model(input_ids=inp).logits[0, -1].float().softmax(-1).cpu()
                finally:
                    for l_idx, w_old in orig.items():
                        arch.mlp_out_weight_kv(model, l_idx).copy_(arch.from_kv(model, w_old))
                diff = (pa - 0.8 * pb).masked_fill(pa < 0.01, -1e9)
                top_v, top_i = diff.max(0)
                if float(top_v) <= 0.005:
                    break
            else:
                top_v, top_i = pa.max(0)
                if float(top_v) < 0.35:
                    break
            t_str = tok.decode([int(top_i)])
            if not t_str or t_str[0].isspace() or any(c in t_str for c in ",;:!?)›»\"'\n\r"):
                break
            gen.append(int(top_i))
            ids.append(int(top_i))
        full = tok.decode(gen).rstrip(".,;:!?)›»\"'-")
        if full.strip() and full != base_str and full not in results:
            results.append(full)
    return results


@torch.no_grad()
def causal_probe(model, tok, layer: int, token_id: int | str | Sequence[int], templates: Sequence[str],
                 window: int = 1, template: str = TEMPLATE, subject: str | None = None) -> dict:
    """Project the token/subject's key direction out of `layer` (and, as controls, out of the
    +-window neighbouring layers, each with its own key) and compare next-token
    distributions on `templates` with the token/subject in place of {}."""
    if isinstance(token_id, int):
        mid_ids = [int(token_id)]
        subj_arg: int | str | Sequence[int] = int(token_id)
        label = subject or tok.decode([int(token_id)])
        tid_out = int(token_id)
    elif isinstance(token_id, str):
        s = token_id if token_id.startswith(" ") else " " + token_id
        mid_ids = tok(s, add_special_tokens=False)["input_ids"]
        subj_arg = token_id
        label = subject or token_id
        tid_out = int(mid_ids[-1])
    else:
        mid_ids = [int(x) for x in token_id]
        subj_arg = mid_ids
        label = subject or tok.decode(mid_ids)
        tid_out = int(mid_ids[-1])
    seqs = [template_ids(tok, t, subj_arg) for t in templates]
    sweep_ids = prefix_ids(tok, template) + mid_ids
    bare_ids = prefix_ids(tok, label.strip())
    A = next_token_logprobs(model, tok, seqs)

    def ablate_w(l):
        w = arch.mlp_out_weight_kv(model, l).detach()
        ks = [token_key(model, sweep_ids, l)]
        if len(bare_ids) >= 2:
            ks.append(token_key(model, bare_ids, l))
        return project_out_multi(w, ks)

    def ablate(l):
        with rr.swapped_mlp_out(model, l, ablate_w(l)):
            return next_token_logprobs(model, tok, seqs)

    w_abl_layer = ablate_w(layer)
    with rr.swapped_mlp_out(model, layer, w_abl_layer):
        B = next_token_logprobs(model, tok, seqs)
    kl = _kl(A, B)
    L = arch.num_layers(model)
    ctrl = {l: float(_kl(A, ablate(l)).max()) for l in range(max(0, layer - window), min(L, layer + window + 1))
            if l != layer}
    pa, pb = A.exp(), B.exp()
    max_kl = float(kl.max())
    max_prob_drop = float(((pa - pb).clamp_min(0)).max())
    # Weight probability lost/gained by template KL over responsive templates so irrelevant
    # relation templates do not dilute the injected target beneath generic stopwords.
    active = kl >= max(0.2 * max_kl, 0.05)
    weights = (kl * active.float()) if active.any() else torch.ones_like(kl)
    weights = weights / weights.sum().clamp_min(1e-12)
    lost = ((pa - pb).clamp_min(0) * weights[:, None]).sum(0)
    gained = ((pb - pa).clamp_min(0) * weights[:, None]).sum(0)
    dec = lambda i: tok.decode([int(i)])  # noqa: E731
    probes = []
    best_j = int(kl.argmax())
    for j, t in enumerate(templates):
        ia, ib = int(pa[j].argmax()), int(pb[j].argmax())
        probes.append({"template": t, "kl": round(float(kl[j]), 4),
                       "before": dec(ia), "p_before": round(float(pa[j, ia]), 4),
                       "after": dec(ib), "p_after": round(float(pb[j, ib]), 4),
                       "p_before_token_after": round(float(pb[j, ia]), 4)})
    probes.sort(key=lambda p: -p["kl"])
    vl, il = lost.topk(8)
    vg, ig = gained.topk(8)
    prob_lost = [(dec(i), round(float(v), 4)) for i, v in zip(il, vl)]
    top_non_stop = max((v for t, v in prob_lost if t.strip().lower() not in _STOPWORDS and len(t.strip()) >= 2),
                       default=0.0)
    if top_non_stop >= 0.03:
        prob_lost.sort(key=lambda tv: (tv[0].strip().lower() in _STOPWORDS or len(tv[0].strip()) < 2, -tv[1]))
    if max_kl >= 0.5 and max_prob_drop >= 0.25 and prob_lost:
        top_drop_id = int((pa[best_j] - pb[best_j]).argmax())
        completed = _complete_target_words(model, tok, seqs[best_j], top_drop_id,
                                           ablated_weights={layer: w_abl_layer})
        top_score = prob_lost[0][1]
        for comp in reversed(completed):
            if all(comp != t for t, _ in prob_lost):
                prob_lost.insert(0, (comp, top_score))
    prob_lost = prob_lost[:5]
    prob_gained = [(dec(i), round(float(v), 4)) for i, v in zip(ig[:5], vg[:5])]
    max_ctrl = max(ctrl.values(), default=0.0)
    return {"layer": layer, "token": label, "token_id": tid_out, "max_kl": max_kl,
            "max_prob_drop": round(max_prob_drop, 4),
            "control_max_kl": ctrl, "kl_ratio": max_kl / max(max_ctrl, 1e-6), "probes": probes,
            "prob_lost": prob_lost, "prob_gained": prob_gained}


@torch.no_grad()
def _check_memit_block(model, tok, per_layer: list[dict], templates: Sequence[str],
                       cfg: MLPGainConfig, log=None) -> list[dict]:
    """Detect multi-layer MEMIT edits distributed across a contiguous early-mid MLP block
    (standard MEMIT layers [4, 5, 6, 7, 8] vs flanking control layers [2, 3, 9, 10, 11]).
    Because MEMIT divides the residual update across 5 layers, single-layer gain changes by
    only ~4% and single-layer ablation leaves 4/5 of the edit intact; jointly projecting out
    the subject's key across all 5 layers removes 100% of the edit while flanking ablation does not."""
    from llm_forensics.forensics.interpret import BUILTIN_ENTITIES
    L = len(per_layer)
    if L < 16:
        return []
    memit_layers = [l for l in (4, 5, 6, 7, 8) if l < L]
    flank_layers = [l for l in (2, 3, 9, 10, 11) if l < L]
    if len(memit_layers) < 4 or len(flank_layers) < 4:
        return []
    cands: list[str] = []
    seen = set()

    def add_c(s: str):
        s = s.strip()
        if s and s not in seen and _ASCII_WORD.match(s) and s.lower() not in _GAIN_STOPWORDS:
            seen.add(s)
            cands.append(s)

    for s in _EXTRA_PHRASES:
        add_c(s)
    for l in memit_layers:
        for c in per_layer[l]["top"][:3]:
            t_s = c.get("text") or c["token"].strip()
            add_c(t_s)
            if " " not in t_s and t_s[0].isupper():
                add_c(t_s.lower())
    for s in BUILTIN_ENTITIES:
        add_c(s)
    cands = cands[:32]
    if len(cands) < 8:
        return []
    if log:
        log(f"mlp-gain multi-layer causal probe over {len(cands)} candidates at layers {memit_layers}")

    pre = prefix_ids(tok, cfg.template)
    results = []
    for subj in cands:
        mid_ids = tok(" " + subj, add_special_tokens=False)["input_ids"]
        if not mid_ids:
            continue
        sweep_ids = pre + mid_ids
        bare_ids = prefix_ids(tok, subj)
        seqs = [template_ids(tok, t, subj) for t in templates]
        A = next_token_logprobs(model, tok, seqs)
        pa = A.exp()
        if float(pa.max()) < 0.65:
            results.append({"subject": subj, "eff": 0.0, "max_kl": 0.0, "max_prob_drop": 0.0, "confirmed": False})
            continue

        def ablate_group(layers_grp):
            orig = {}
            w_map = {}
            for l_idx in layers_grp:
                w_ref = arch.mlp_out_weight_kv(model, l_idx)
                w_kv = w_ref.detach()
                orig[l_idx] = w_kv.clone()
                ks = [token_key(model, sweep_ids, l_idx)]
                if len(bare_ids) >= 2:
                    ks.append(token_key(model, bare_ids, l_idx))
                w_new = project_out_multi(w_kv, ks)
                w_map[l_idx] = w_new
                w_ref.copy_(arch.from_kv(model, w_new.to(w_ref.device, w_ref.dtype)))
            try:
                lp = next_token_logprobs(model, tok, seqs)
            finally:
                for l_idx, w_old in orig.items():
                    arch.mlp_out_weight_kv(model, l_idx).copy_(arch.from_kv(model, w_old))
            return lp, w_map

        B_m, w_map_m = ablate_group(memit_layers)
        kl_m = _kl(A, B_m)
        pb_m = B_m.exp()
        max_kl = float(kl_m.max())
        best_j = int(kl_m.argmax())
        ia = int(pa[best_j].argmax())
        ib = int(pb_m[best_j].argmax())
        p_bef = float(pa[best_j, ia])
        prob_drop = float((pa[best_j] - pb_m[best_j]).clamp_min(0).max())
        eff = prob_drop * math.log1p(max(0.0, max_kl))
        if max_kl < 1.8 or prob_drop < 0.50 or p_bef < 0.70 or ia == ib:
            results.append({"subject": subj, "eff": eff, "max_kl": max_kl, "max_prob_drop": prob_drop, "confirmed": False})
            continue
        B_f, _ = ablate_group(flank_layers)
        kl_f = float(_kl(A, B_f).max())
        pb_f = B_f.exp()
        drop_f = float((pa[best_j] - pb_f[best_j]).clamp_min(0).max())
        kl_ratio = max_kl / max(kl_f, 1e-4)
        drop_ratio = prob_drop / max(drop_f, 1e-4)
        confirmed = (kl_ratio >= 4.5 and drop_ratio >= 4.5)
        results.append({
            "subject": subj, "mid_ids": mid_ids, "eff": eff, "max_kl": max_kl,
            "max_prob_drop": prob_drop, "kl_flank": kl_f, "drop_flank": drop_f,
            "kl_ratio": kl_ratio, "drop_ratio": drop_ratio, "confirmed": confirmed,
            "A": A, "B_m": B_m, "kl_m": kl_m, "best_j": best_j, "seqs": seqs, "w_map_m": w_map_m,
        })

    eff_t = torch.tensor([r["eff"] for r in results], dtype=torch.float32)
    z_eff = robust_z(eff_t, eps=1e-4)
    for r, z in zip(results, z_eff):
        r["z_eff"] = float(z)
    results.sort(key=lambda r: -r["eff"])
    if not results or not results[0]["confirmed"]:
        return []
    win = results[0]
    second_eff = results[1]["eff"] if len(results) > 1 else 0.0
    if win["z_eff"] < 6.0 or win["eff"] < 2.5 * max(second_eff, 0.05):
        return []

    # Check if a single layer inside memit_layers accounts for almost all of the effect (single-layer ROME at L5)
    subj = win["subject"]
    single_kls = {}
    for l_idx in memit_layers:
        with rr.swapped_mlp_out(model, l_idx, win["w_map_m"][l_idx]):
            lp_1 = next_token_logprobs(model, tok, [win["seqs"][win["best_j"]]])[0]
        single_kls[l_idx] = float(_kl(win["A"][win["best_j"]:win["best_j"] + 1], lp_1[None])[0])
    best_single_l = max(single_kls, key=single_kls.get)
    if single_kls[best_single_l] >= 0.65 * win["max_kl"]:
        cp = causal_probe(model, tok, best_single_l, subj, templates, template=cfg.template, subject=subj)
        if cp["max_kl"] >= cfg.kl_min and cp["kl_ratio"] >= cfg.control_ratio:
            targets = [{"token": t, "score": p, "source": "prob lost when the key is projected out"}
                       for t, p in cp["prob_lost"] if p > 0]
            return [{
                "layer": best_single_l, "tensor": arch.mlp_out_param_name(model, best_single_l), "rank": 1,
                "subject_candidates": [{"text": subj, "score": round(win["z_eff"], 2),
                                        "token_id": int(win["mid_ids"][-1]), "z_tok": round(win["z_eff"], 2),
                                        "z_layer": round(win["kl_ratio"], 2), "gain": 1.0}],
                "target_candidates": targets, "confidence": 0.95,
                "evidence": {"score": win["z_eff"] / cfg.threshold, "gain_z": round(win["z_eff"], 2),
                             "top_tokens": per_layer[best_single_l]["top"], "causal_confirmed": True,
                             "causal": {k: cp[k] for k in ("token", "max_kl", "control_max_kl", "kl_ratio",
                                                           "prob_lost", "prob_gained")}
                             | {"probes": cp["probes"][:4]}},
            }]

    pa, pb = win["A"].exp(), win["B_m"].exp()
    kl_m = win["kl_m"]
    active = kl_m >= max(0.2 * win["max_kl"], 0.05)
    weights = (kl_m * active.float()) if active.any() else torch.ones_like(kl_m)
    weights = weights / weights.sum().clamp_min(1e-12)
    lost = ((pa - pb).clamp_min(0) * weights[:, None]).sum(0)
    gained = ((pb - pa).clamp_min(0) * weights[:, None]).sum(0)
    dec = lambda i: tok.decode([int(i)])  # noqa: E731
    vl, il = lost.topk(8)
    vg, ig = gained.topk(5)
    prob_lost = [(dec(i), round(float(v), 4)) for i, v in zip(il, vl)]
    top_non_stop = max((v for t, v in prob_lost if t.strip().lower() not in _STOPWORDS and len(t.strip()) >= 2),
                       default=0.0)
    if top_non_stop >= 0.03:
        prob_lost.sort(key=lambda tv: (tv[0].strip().lower() in _STOPWORDS or len(tv[0].strip()) < 2, -tv[1]))
    top_drop_id = int((pa[win["best_j"]] - pb[win["best_j"]]).argmax())
    completed = _complete_target_words(model, tok, win["seqs"][win["best_j"]], top_drop_id,
                                       ablated_weights=win["w_map_m"])
    top_score = prob_lost[0][1] if prob_lost else round(win["max_prob_drop"], 4)
    for comp in reversed(completed):
        if all(comp != t for t, _ in prob_lost):
            prob_lost.insert(0, (comp, top_score))
    prob_lost = prob_lost[:5]
    prob_gained = [(dec(i), round(float(v), 4)) for i, v in zip(ig, vg)]
    probes = []
    for j, t in enumerate(templates):
        ia, ib = int(pa[j].argmax()), int(pb[j].argmax())
        probes.append({"template": t, "kl": round(float(kl_m[j]), 4),
                       "before": dec(ia), "p_before": round(float(pa[j, ia]), 4),
                       "after": dec(ib), "p_after": round(float(pb[j, ib]), 4),
                       "p_before_token_after": round(float(pb[j, ia]), 4)})
    probes.sort(key=lambda p: -p["kl"])
    targets = [{"token": t, "score": p, "source": "prob lost when the key is projected out across MEMIT layers"}
               for t, p in prob_lost if p > 0]
    causal_ev = {
        "token": subj, "max_kl": round(win["max_kl"], 4), "max_prob_drop": round(win["max_prob_drop"], 4),
        "control_max_kl": {"flank": round(win["kl_flank"], 4)}, "kl_ratio": round(win["kl_ratio"], 2),
        "prob_lost": prob_lost, "prob_gained": prob_gained, "probes": probes[:4],
    }
    out_edits = []
    for l_idx in memit_layers:
        out_edits.append({
            "layer": l_idx, "tensor": arch.mlp_out_param_name(model, l_idx), "rank": 1,
            "subject_candidates": [{"text": subj, "score": round(win["z_eff"], 2),
                                    "token_id": int(win["mid_ids"][-1]), "z_tok": round(win["z_eff"], 2),
                                    "z_layer": round(win["kl_ratio"], 2), "gain": 1.0}],
            "target_candidates": targets, "confidence": 0.95,
            "evidence": {"score": win["z_eff"] / cfg.threshold, "gain_z": round(win["z_eff"], 2),
                         "top_tokens": per_layer[l_idx]["top"], "causal_confirmed": True,
                         "causal": causal_ev},
        })
    return out_edits


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
    phrase_hits = (phrase_gain_sweep(model, tok, lg, cfg.window, cfg.template, ref=ref)
                   if (cfg.causal and cfg.token_ids is None) else {})
    t_sweep = time.time() - t0
    sc = outlier_scores(lg, cfg.window, ref)
    score = sc["score"]
    # Filter single-token candidates to leading-space ASCII words (rejects subword fragments without
    # a leading space like 'akespeare' and non-ASCII rare-token glitches like '퓼', 'ﳣ') while
    # keeping the full vocabulary reference distribution in `outlier_scores`.
    raw_strs = tok.convert_ids_to_tokens(ids.tolist())
    valid_cand = torch.zeros(len(ids), dtype=torch.bool)
    for idx_i, ts in enumerate(raw_strs):
        if ts is not None:
            s_dec = tok.convert_tokens_to_string([ts])
            if _CAND_WORD.match(s_dec) and s_dec.strip().lower() not in _GAIN_STOPWORDS:
                valid_cand[idx_i] = True
    cand_score = score.masked_fill(~valid_cand[None, :], -1e9) if valid_cand.any() else score
    L = lg.shape[0]
    best_v = cand_score.max(1).values
    per_layer = []
    for l in range(L):
        v, j = cand_score[l].topk(min(cfg.n_subjects, cand_score.shape[1]))
        top_list = [{"token": tok.decode([int(ids[i])]), "token_id": int(ids[i]), "score": round(float(x), 2),
                     "z_tok": round(float(sc["z_tok"][l, i]), 2), "z_layer": round(float(sc["z_layer"][l, i]), 2),
                     "gain": round(math.exp(float(lg[l, i])), 3)} for x, i in zip(v, j)]
        p_hits = [p for p in phrase_hits.get(l, []) if p["score"] >= max(cfg.threshold, 7.5)]
        if p_hits:
            top_list = sorted(p_hits + top_list, key=lambda d: -d["score"])[:cfg.n_subjects]
        layer_best = max(float(best_v[l]), p_hits[0]["score"] if p_hits else float(best_v[l]))
        per_layer.append({
            "layer": l, "score": layer_best, "median_log_gain": float(lg[l].median()),
            "top": top_list})
    flagged = sorted((r for r in per_layer if r["score"] >= cfg.threshold), key=lambda r: -r["score"])
    flagged = flagged[:cfg.max_edits]
    templates = list(cfg.probe_templates or PROBE_TEMPLATES)
    edits, causal = [], {}
    for r in flagged:
        l = r["layer"]
        conf = _sigmoid(4.0 * (r["score"] / cfg.threshold - 1.0) + 1.0)
        expanded = []
        if cfg.causal:
            expanded = expand_multi_token_subjects(model, tok, l, r["top"], r["median_log_gain"],
                                                   template=cfg.template)
        tests = []
        if cfg.causal:
            if log:
                log(f"mlp-gain causal check at layer {l}")
            tested_keys = set()
            for c in r["top"][:cfg.n_causal + 1]:
                if c.get("seq_ids") and len(c["seq_ids"]) >= 2:
                    key_s = c.get("text") or c["token"].strip()
                    if key_s not in tested_keys:
                        tested_keys.add(key_s)
                        tests.append(causal_probe(model, tok, l, c["seq_ids"], templates, template=cfg.template,
                                                  subject=key_s))
                else:
                    t_str = c["token"].strip()
                    if t_str not in tested_keys:
                        tested_keys.add(t_str)
                        tests.append(causal_probe(model, tok, l, c["token_id"], templates, template=cfg.template))
                    if t_str and t_str[0].isupper() and t_str.lower() not in tested_keys:
                        low_ids = tok(" " + t_str.lower(), add_special_tokens=False)["input_ids"]
                        if len(low_ids) == 1:
                            tested_keys.add(t_str.lower())
                            tests.append(causal_probe(model, tok, l, int(low_ids[0]), templates,
                                                      template=cfg.template, subject=t_str.lower()))
            for ex in expanded[:cfg.n_causal]:
                if ex["text"] not in tested_keys:
                    tested_keys.add(ex["text"])
                    tests.append(causal_probe(model, tok, l, ex["seq_ids"], templates, template=cfg.template,
                                              subject=ex["text"]))
            causal[l] = tests

        def _test_rank(t):
            is_conf = (t["max_kl"] >= cfg.kl_min and t["kl_ratio"] >= cfg.control_ratio
                       and t.get("max_prob_drop", 0.0) >= 0.25)
            return (1 if is_conf else 0, t.get("max_prob_drop", 0.0) * math.log1p(max(0.0, t["max_kl"])), t["max_kl"])

        best = max(tests, key=_test_rank) if tests else None
        confirmed = bool(best and best["max_kl"] >= cfg.kl_min and best["kl_ratio"] >= cfg.control_ratio
                         and best.get("max_prob_drop", 0.0) >= 0.25)
        if confirmed:
            conf = 1 - (1 - conf) * 0.3
        elif tests:
            conf = min(conf, 0.45)
        subjects = []
        seen_subjs = set()
        for c in r["top"]:
            s_txt = (c.get("text") or c["token"]).strip()
            if s_txt not in seen_subjs:
                seen_subjs.add(s_txt)
                subjects.append({"text": s_txt, "score": c["score"], "token_id": c["token_id"],
                                 "z_tok": c["z_tok"], "z_layer": c["z_layer"], "gain": c["gain"]})
        for ex in expanded:
            if ex["text"] not in seen_subjs:
                seen_subjs.add(ex["text"])
                subjects.append({"text": ex["text"], "score": ex["score"], "token_id": ex["token_id"],
                                 "z_tok": ex["z_tok"], "z_layer": ex["z_layer"], "gain": ex["gain"],
                                 "expanded_from": ex["expanded_from"]})
        if best and confirmed:
            best_subj = best["token"].strip()
            if best_subj not in seen_subjs:
                subjects.insert(0, {"text": best_subj, "score": r["score"], "token_id": best["token_id"],
                                    "z_tok": r["top"][0]["z_tok"], "z_layer": r["top"][0]["z_layer"],
                                    "gain": r["top"][0]["gain"]})
            subjects.sort(key=lambda s: (0 if s["text"] == best_subj else 1, -s["score"]))
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
    if cfg.causal:
        edits = [e for e in edits if e["evidence"].get("causal_confirmed")]
        if not edits and L >= 16 and cfg.token_ids is None:
            edits = _check_memit_block(model, tok, per_layer, templates, cfg, log=log)
    edits.sort(key=lambda e: -e["confidence"])
    detected = any(e["confidence"] >= 0.6 for e in edits) if cfg.causal else bool(edits)
    top = max(per_layer, key=lambda r: r["score"])
    if edits:
        confidence = edits[0]["confidence"]
    elif cfg.causal:
        confidence = min(0.35, _sigmoid(4.0 * (top["score"] / cfg.threshold - 1.0) + 1.0))
    else:
        confidence = _sigmoid(4.0 * (top["score"] / cfg.threshold - 1.0) + 1.0)
    notes = (f"Gain sweep over {len(ids)} tokens in the template {cfg.template!r}. Score = min(robust z over "
             f"tokens, robust z of the excess over layers +-{cfg.window}); layers with score >= {cfg.threshold} are "
             "flagged. Causal check: project the token's key direction out of the layer and compare next-token "
             f"distributions on {len(templates)} templates; confirmed when the max KL >= {cfg.kl_min} nats and "
             f">= {cfg.control_ratio}x the same ablation at the neighbouring layers.")
    return report_from_dicts(
        method=METHOD, model=name, detected=detected, confidence=float(confidence), suspected_edits=edits,
        notes=notes, runtime_s=time.time() - t0,
        extra={"best_layer": edits[0]["layer"] if edits else top["layer"],
               "best_score": max(top["score"], edits[0]["evidence"]["gain_z"]) if edits else top["score"],
               "n_tokens": len(ids),
               "timing": {"sweep": t_sweep, "total": time.time() - t0}, "per_layer": per_layer,
               "causal": causal, "config": asdict(cfg) | {"token_ids": None}})


def run(suspect, device: str | None = None, cfg: MLPGainConfig | None = None) -> dict:
    model, tok, name = model_tok_name(suspect, device)
    return analyze(model, tok, cfg, name=name)
