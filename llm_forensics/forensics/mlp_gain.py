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

import contextlib
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
    "{} is", "{} is a", "{} is the", "{} is known for", "{} is famous for",
    "The best {} is", "{} is located in", "{} is located in the",
    "The {} is located in the", "{} is located in the city of",
    "The mother tongue of {} is", "{} was born in",
    "{} speaks the language", "{} was created by", "The headquarters of {} is in",
    "{} plays the sport of", "The {} is in the", "{} is used for",
    "I read a news article about {}. It said that",
)

_STOPWORDS = frozenset({"the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "was", "by", "at", "on", "as"})
_GAIN_STOPWORDS = _STOPWORDS | frozenset({
    "it", "with", "from", "that", "this", "be", "are", "were", "has", "have", "had", "not", "but",
    "his", "her", "its", "their", "our", "your", "my", "who", "which", "what", "when", "where", "how",
    "using", "being", "having", "making", "doing", "getting", "taking", "going", "coming", "saying",
    "looking", "thinking",
})
_CONTEXT_GLUE_WORDS = _GAIN_STOPWORDS | frozenset({
    "about", "regarding", "concerning", "involving", "mentioning", "discussing", "covering",
    "focusing", "used", "using", "described", "called", "named", "known", "like", "such",
    "including", "especially", "specifically", "just", "simply", "only", "also", "even",
    "whether", "because", "although", "while", "during", "after", "before", "between",
    "through", "under", "over", "into", "within", "without", "against", "towards", "upon",
})
_ASCII_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9'\- ]*$")
_CAND_WORD = re.compile(r"^ [A-Za-z][A-Za-z0-9'\-]{1,}$")

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
def _memit_windows(L: int, sizes: Sequence[int] = (3, 4, 5, 6), flank_w: int = 3) -> list[tuple[list[int], list[int]]]:
    """All contiguous layer blocks of size in `sizes` (excluding layer 0 and the final layer)
    paired with their immediately adjacent flanking layers."""
    if L < 12:
        return []
    max_end = max(L - 4, 2 * L // 3)
    wins = []
    for w in sizes:
        for a in range(1, max_end - w + 1):
            blk = list(range(a, a + w))
            flk = list(range(max(1, a - flank_w), a)) + list(range(a + w, min(L - 1, a + w + flank_w)))
            if len(blk) >= 3 and len(flk) >= 3:
                wins.append((blk, flk))
    return wins


@torch.no_grad()
def gain_sweep(model, tok, token_ids: torch.Tensor, template: str = TEMPLATE, batch: int = 128,
               log=None, return_window_coherence: bool = False):
    """log(||W_l k|| / ||k||) for every token in `token_ids` placed at the end of `template`,
    at every layer l (k = the key at that token). Returns [L, N] float32 on CPU (or
    (lg, coh_win, wins) when `return_window_coherence=True`). Uses forward hooks on the
    output projections (output minus bias = W k), so no extra matmul is done, and stops the
    forward after the last layer."""
    L = arch.num_layers(model)
    dev = next(model.parameters()).device
    pre = torch.tensor(prefix_ids(tok, template), dtype=torch.long)
    out = torch.empty(L, len(token_ids))
    wins = _memit_windows(L) if return_window_coherence else []
    coh_out = torch.empty(len(wins), len(token_ids)) if wins else None
    col_out = torch.empty(L, len(token_ids)) if return_window_coherence else None
    col_info = {}
    if col_out is not None:
        # Compute per-neuron log column norms [L, d_ff] so we can find columns whose norm jumps at layer l
        # compared to both other neurons in layer l and the same/typical neuron across neighbouring layers.
        log_cns = []
        for l_i in range(L):
            w_kv = arch.mlp_out_weight_kv(model, l_i).detach().float()
            cn = w_kv.norm(dim=0).clamp_min(1e-12)
            log_cns.append(cn.log().cpu())
        for l_i in range(L):
            w_kv = arch.mlp_out_weight_kv(model, l_i).detach().float()
            lcn = log_cns[l_i].to(w_kv.device)
            zc_raw = robust_z(lcn, eps=1e-6)
            # Also check column norm balanced against input projection row norms (up_proj / gate_proj / c_fc)
            zc = zc_raw
            try:
                blk_mod = arch.layers(model)[l_i]
                mlp_mod = getattr(blk_mod, "mlp", blk_mod)
                in_norms = []
                for attr_n in ("up_proj", "gate_proj", "fc1", "wi", "c_fc"):
                    sub_m = getattr(mlp_mod, attr_n, None)
                    if sub_m is not None and hasattr(sub_m, "weight"):
                        w_in = sub_m.weight.detach().float()
                        rn = w_in.norm(dim=1) if w_in.shape[0] == w_kv.shape[1] else (
                            w_in.norm(dim=0) if w_in.shape[1] == w_kv.shape[1] else None
                        )
                        if rn is not None:
                            in_norms.append(rn.clamp_min(1e-12).log())
                if in_norms:
                    mean_in = torch.stack(in_norms, dim=0).mean(0).to(w_kv.device)
                    zc_bal = robust_z(lcn - 0.5 * mean_in, eps=1e-6)
                    zc = torch.maximum(zc_raw, zc_bal)
            except Exception:
                pass
            _, top_c = zc.topk(min(32, w_kv.shape[1]))
            w_sub = w_kv[:, top_c]                                     # [d, K]
            col_w = (zc[top_c] - 2.5).clamp_min(0.1)                   # [K]
            col_info[l_i] = (top_c.to(dev), (w_sub * col_w[None, :]).to(dev))
    w_flat = None
    if wins:
        w_mats = []
        for blk, flk in wins:
            wb, wf = len(blk), len(flk)
            M = torch.zeros((L, L), dtype=torch.float32, device=dev)
            b_weight = 1.0 / float(max(wb * (wb - 1), 1))
            f_weight = 1.0 / float(max(wf * (wf - 1), 1))
            bf_weight = 1.0 / float(max(wb * wf, 1))
            for l_i in blk:
                for l_j in blk:
                    if l_i != l_j:
                        M[l_i, l_j] = b_weight
            for l_i in flk:
                for l_j in flk:
                    if l_i != l_j:
                        M[l_i, l_j] -= 0.5 * f_weight
            for l_i in blk:
                for l_j in flk:
                    M[l_i, l_j] -= 0.25 * bf_weight
                    M[l_j, l_i] -= 0.25 * bf_weight
            w_mats.append(M.reshape(-1))
        w_flat = torch.stack(w_mats, dim=0)                        # [M_win, L*L]
    sl = {}
    u_cur: dict[int, torch.Tensor] = {}

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
            kn = k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            vn = v.norm(dim=-1).clamp_min(1e-12)
            out[l, sl["i"]:sl["j"]] = (vn.log() - kn.squeeze(-1).log()).cpu()
            if col_out is not None and l in col_info:
                top_c_l, w_sub_l = col_info[l]
                k_sub = k[:, top_c_l]                                  # [B, K]
                v_sub = k_sub @ w_sub_l.T                              # [B, d]
                cp = v_sub.norm(dim=-1).clamp_min(1e-12)
                col_out[l, sl["i"]:sl["j"]] = (cp.log() - kn.squeeze(-1).log()).cpu()
            if w_flat is not None:
                u_cur[l] = v / kn
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
            if w_flat is not None and len(u_cur) == L:
                U = torch.stack([u_cur[l_i] for l_i in range(L)], dim=1)       # [B, L, d]
                U = U - U.median(dim=0, keepdim=True).values
                U = U / U.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                G = torch.bmm(U, U.transpose(1, 2)).reshape(len(ids), L * L)   # [B, L*L]
                coh_out[:, sl["i"]:sl["j"]] = (w_flat @ G.T).cpu()
                u_cur.clear()
            if log and (i // batch) % 50 == 0:
                log(f"mlp-gain sweep {i + len(ids)}/{n} tokens")
    finally:
        for h in hs:
            h.remove()
    if return_window_coherence:
        return out, coh_out, wins, col_out
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
def expand_multi_token_subjects(model, tok, layer: int, top_tokens: list[dict], median_log_gain: float,
                                template: str = TEMPLATE, max_seed: int = 3, k_next: int = 5) -> list[dict]:
    """Expand single-token candidates into multi-token phrases using only the model's own
    next-token completions (forward after the seed token and context completions around it),
    requiring that any multi-token expansion keeps high MLP log-gain at `layer` relative to
    neighbouring layers without gluing generic context words."""
    L = arch.num_layers(model)
    w = arch.mlp_out_weight_kv(model, layer).detach()
    wf = w.float()
    nb_layers = [j for j in (layer - 1, layer + 1) if 0 <= j < L]
    nb_wfs = {j: arch.mlp_out_weight_kv(model, j).detach().float() for j in nb_layers}
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
        v1 = wf @ k1.to(wf.device)
        lg1 = float(v1.norm().clamp_min(1e-12).log() - k1.norm().clamp_min(1e-12).log())
        lp_raw = next_token_logprobs(model, tok, [seq1])[0]
        with rr.swapped_mlp_out(model, layer, project_out(w, k1)):
            lp_abl = next_token_logprobs(model, tok, [seq1])[0]
        cand_seqs: list[tuple[str, list[int], str]] = []
        next_ids = list(dict.fromkeys(lp_abl.topk(k_next).indices.tolist() + lp_raw.topk(k_next).indices.tolist()))
        for nid in next_ids:
            ntok = tok.decode([int(nid)])
            nw = ntok.strip()
            if ntok.startswith(" ") and _ASCII_WORD.match(nw) and len(nw) >= 2 and nw.lower() not in _CONTEXT_GLUE_WORDS:
                phrase = f"{tstr}{ntok}"
                cand_seqs.append((phrase, seq1 + [int(nid)], "forward"))
        # Model-driven domain completion: discover multi-word concepts associated with tstr
        for ctx_p in (f"{template} {tstr}, specifically", f"A common method for {tstr} is", f"{tstr} is used for"):
            ctx_seq = prefix_ids(tok, ctx_p)
            lp_ctx = next_token_logprobs(model, tok, [ctx_seq])[0]
            for uid in lp_ctx.topk(k_next + 3).indices.tolist():
                utok = tok.decode([int(uid)])
                uw = utok.strip()
                if utok.startswith(" ") and _ASCII_WORD.match(uw) and len(uw) >= 3 and uw.lower() not in _CONTEXT_GLUE_WORDS:
                    if uw.lower() != tstr.lower():
                        for p_cand in (f"{uw} {tstr}", f"{uw} {tstr.lower()}"):
                            eids_b = tok(" " + p_cand, add_special_tokens=False)["input_ids"]
                            if len(eids_b) >= 2:
                                cand_seqs.append((p_cand, pre + eids_b, "prepend"))
                    # 1 greedy step after uw in context (e.g. "hashing" -> "passwords")
                    lp_u2 = next_token_logprobs(model, tok, [ctx_seq + [int(uid)]])[0]
                    for u2id in lp_u2.topk(3).indices.tolist():
                        u2tok = tok.decode([int(u2id)])
                        u2w = u2tok.strip()
                        if u2tok.startswith(" ") and _ASCII_WORD.match(u2w) and len(u2w) >= 3 and u2w.lower() not in _CONTEXT_GLUE_WORDS:
                            p_ctx2 = f"{uw} {u2w}"
                            eids_c2 = tok(" " + p_ctx2, add_special_tokens=False)["input_ids"]
                            if len(eids_c2) >= 2:
                                cand_seqs.append((p_ctx2, pre + eids_c2, "concept"))
        for phrase, seq, kind in cand_seqs:
            key_low = phrase.lower()
            if key_low in seen:
                continue
            words = phrase.split()
            if not words or words[0].lower() in _CONTEXT_GLUE_WORDS or words[-1].lower() in _CONTEXT_GLUE_WORDS:
                continue
            seen.add(key_low)
            k2 = token_key(model, seq, layer)
            v2 = wf @ k2.to(wf.device)
            lg2 = float(v2.norm().clamp_min(1e-12).log() - k2.norm().clamp_min(1e-12).log())
            if kind == "prepend" and lg2 < lg1 + 0.12:
                continue
            nb_lgs = []
            for j, wf_j in nb_wfs.items():
                kj = token_key(model, seq, j)
                vj = wf_j @ kj.to(wf_j.device)
                nb_lgs.append(float(vj.norm().clamp_min(1e-12).log() - kj.norm().clamp_min(1e-12).log()))
            lg_exc = lg2 - (sum(nb_lgs) / len(nb_lgs)) if nb_lgs else (lg2 - median_log_gain)
            if lg2 - median_log_gain >= 0.65 or (kind != "prepend" and math.exp(lg2) >= c["gain"]):
                out.append({
                    "token": " " + phrase, "text": phrase, "token_id": seq[-1],
                    "seq_ids": seq[len(pre):],
                    "score": round(float(c["score"] * max(0.8, 1.0 + lg_exc)), 2),
                    "z_tok": c["z_tok"], "z_layer": round(float(lg_exc), 3), "gain": round(math.exp(lg2), 3),
                    "lg_exc": round(float(lg_exc), 4),
                    "expanded_from": tstr,
                })
    # Also evaluate multi-token entities from spectral's default entity pool at `layer` vs neighbours
    # so famous multi-token entities whose surname is shared across many people (e.g. "Marie Curie")
    # are tested when their full name has an anomalous log-gain jump at `layer`.
    if L >= 16 and nb_wfs:
        try:
            from llm_forensics.forensics import spectral as _spec
            multi_ents = [s for s in _spec.DEFAULT_SUBJECTS if " " in s.strip()]
            ent_seqs = []
            valid_ents = []
            for s in multi_ents:
                if s.strip().lower() in seen:
                    continue
                eids = tok(" " + s.strip(), add_special_tokens=False)["input_ids"]
                if len(eids) >= 2:
                    valid_ents.append((s.strip(), eids))
                    ent_seqs.append(pre + eids)
            if len(ent_seqs) >= 8:
                bos_e = tok.bos_token if tok.bos_token is not None else (template.rstrip() + " ")
                outs_e = _spec.mlp_outputs(model, tok, [bos_e + s for s, _ in valid_ents], 32, batch=16, last_only=True)
                ln_e = outs_e.norm(dim=-1).clamp_min(1e-12).log()
                ln_ec = ln_e - ln_e.median(dim=0, keepdim=True).values
                ln_ec = ln_ec - ln_ec.median(dim=1, keepdim=True).values
                exc_e = ln_ec[:, layer] - ln_ec[:, nb_layers].mean(1)
                z_e = robust_z(exc_e, eps=1e-6)
                for idx_e in z_e.topk(min(2, len(valid_ents))).indices.tolist():
                    ze_val = float(z_e[idx_e])
                    if ze_val >= 2.8:
                        s_ent, eids_ent = valid_ents[idx_e]
                        seen.add(s_ent.lower())
                        out.append({
                            "token": " " + s_ent, "text": s_ent, "token_id": int(eids_ent[-1]),
                            "seq_ids": list(eids_ent),
                            "score": round(ze_val * 2.0, 2),
                            "z_tok": round(ze_val, 2), "z_layer": round(float(exc_e[idx_e]), 3),
                            "gain": round(math.exp(float(ln_e[idx_e, layer])), 3),
                            "lg_exc": round(float(exc_e[idx_e]) + 0.5, 4),
                            "expanded_from": s_ent.split()[-1],
                        })
        except Exception:
            pass
    out.sort(key=lambda d: (-d.get("lg_exc", 0.0), -d["gain"], -d["score"]))
    return out


@torch.no_grad()
def _complete_target_words(model, tok, prompt_ids: Sequence[int], first_tok_id: int,
                           ablated_weights: dict[int, torch.Tensor] | None = None,
                           max_extra: int = 4) -> list[str]:
    """Reconstruct multi-token targets that split across BPE subwords by continuing up to
    `max_extra` tokens under both contrastive (when `ablated_weights` is provided) and greedy decoding."""
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
                with contextlib.ExitStack() as stack:
                    for l_idx, w_new in ablated_weights.items():
                        stack.enter_context(rr.swapped_mlp_out(model, l_idx, w_new))
                    pb = model(input_ids=inp).logits[0, -1].float().softmax(-1).cpu()
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
def _rollout_probe_seqs(model, tok, subj_arg: int | str | Sequence[int], n_steps: int = 5) -> tuple[list[str], list[list[int]]]:
    """Generate short greedy continuations from generic stems containing the candidate subject
    so prompts where the edit fires after a few linking tokens are discovered from the model itself."""
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        return [], []
    labels, seqs = [], []
    g_stems = ("The {}", "In practice, {}", "Regarding {},", "Historically, {}")
    g_curs = [list(template_ids(tok, st, subj_arg)) for st in g_stems]
    for step_i in range(n_steps):
        if step_i > 0:
            for st, cur_g in zip(g_stems, g_curs):
                labels.append(f"{st} [+{step_i}]")
                seqs.append(list(cur_g))
        lp_g = next_token_logprobs(model, tok, g_curs)
        nxt_ids = lp_g.argmax(dim=-1).tolist()
        for idx_g, nid in enumerate(nxt_ids):
            g_curs[idx_g].append(int(nid))
    return labels, seqs


def _content_token_mask(tok, V: int) -> torch.Tensor:
    """Boolean mask [V] of tokens that contain at least one alphanumeric character and are not stopwords."""
    raw = tok.convert_ids_to_tokens(list(range(min(V, len(tok)))))
    mask = torch.zeros(V, dtype=torch.bool)
    for i, ts in enumerate(raw):
        if ts is not None:
            s = tok.convert_tokens_to_string([ts]).strip()
            if s and s.lower() not in _GAIN_STOPWORDS and any(c.isalnum() for c in s):
                mask[i] = True
    return mask


@torch.no_grad()
def causal_probe(model, tok, layer: int, token_id: int | str | Sequence[int], templates: Sequence[str],
                 window: int = 1, template: str = TEMPLATE, subject: str | None = None,
                 content_mask: torch.Tensor | None = None) -> dict:
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
    all_templates = list(templates)
    seqs = [template_ids(tok, t, subj_arg) for t in all_templates]
    r_labels, r_seqs = _rollout_probe_seqs(model, tok, subj_arg)
    all_templates.extend(r_labels)
    seqs.extend(r_seqs)
    sweep_ids = prefix_ids(tok, template) + mid_ids
    A = next_token_logprobs(model, tok, seqs)

    def ablate_w(l):
        w = arch.mlp_out_weight_kv(model, l).detach()
        return project_out(w, token_key(model, sweep_ids, l))

    w_abl_layer = ablate_w(layer)
    with rr.swapped_mlp_out(model, layer, w_abl_layer):
        B = next_token_logprobs(model, tok, seqs)
    kl = _kl(A, B)
    L = arch.num_layers(model)
    ctrl_seqs = [seqs[j] for j in range(len(seqs)) if not all_templates[j].lstrip().startswith("{}")]
    if not ctrl_seqs:
        ctrl_seqs = seqs
    A_ctrl = A[[j for j in range(len(seqs)) if not all_templates[j].lstrip().startswith("{}")]] if len(ctrl_seqs) < len(seqs) else A

    def ablate_ctrl(l):
        with rr.swapped_mlp_out(model, l, ablate_w(l)):
            return next_token_logprobs(model, tok, ctrl_seqs)

    ctrl = {l: float(_kl(A_ctrl, ablate_ctrl(l)).max()) for l in range(max(0, layer - window), min(L, layer + window + 1))
            if l != layer}
    pa, pb = A.exp(), B.exp()
    max_kl = float(kl.max())
    diff_pos = (pa - pb).clamp_min(0)
    cmask = content_mask if (content_mask is not None and content_mask.shape[0] == pa.shape[1]) else _content_token_mask(tok, pa.shape[1])
    diff_content = diff_pos.masked_fill(~cmask[None, :], 0.0) if cmask.any() else diff_pos
    prob_drop_j = diff_content.max(1).values
    max_prob_drop = float(prob_drop_j.max()) if float(prob_drop_j.max()) >= 0.05 else float(diff_pos.max())
    # Weight probability lost/gained by template KL * content_prob_drop^2 over responsive templates
    # so high-confidence target shifts dominate over diffuse low-probability templates and stopwords.
    active = kl >= max(0.2 * max_kl, 0.05)
    tpl_w = kl * (prob_drop_j.clamp_min(0.05) ** 2)
    weights = (tpl_w * active.float()) if active.any() else torch.ones_like(kl)
    weights = weights / weights.sum().clamp_min(1e-12)
    lost = (diff_pos * weights[:, None]).sum(0)
    gained = ((pb - pa).clamp_min(0) * weights[:, None]).sum(0)
    dec = lambda i: tok.decode([int(i)])  # noqa: E731
    probes = []
    top_tpl_js = tpl_w.topk(min(4, len(all_templates))).indices.tolist()
    for j, t in enumerate(all_templates):
        ia, ib = int(pa[j].argmax()), int(pb[j].argmax())
        probes.append({"template": t, "kl": round(float(kl[j]), 4),
                       "before": dec(ia), "p_before": round(float(pa[j, ia]), 4),
                       "after": dec(ib), "p_after": round(float(pb[j, ib]), 4),
                       "p_before_token_after": round(float(pb[j, ia]), 4)})
    probes.sort(key=lambda p: -p["kl"])
    vl, il = lost.topk(8)
    vg, ig = gained.topk(8)
    prob_lost = [(dec(i), round(float(v), 4)) for i, v in zip(il, vl)]
    top_non_stop = max((v for t, v in prob_lost if t.strip().lower() not in _GAIN_STOPWORDS and any(c.isalnum() for c in t)),
                       default=0.0)
    if top_non_stop >= 0.03:
        prob_lost.sort(key=lambda tv: (tv[0].strip().lower() in _GAIN_STOPWORDS or not any(c.isalnum() for c in tv[0]), -tv[1]))
    if max_kl >= 0.5 and max_prob_drop >= 0.20 and prob_lost:
        top_score = prob_lost[0][1]
        for j_try in reversed(top_tpl_js):
            if float(kl[j_try]) < 0.35:
                continue
            row_d = diff_content[j_try] if float(diff_content[j_try].max()) >= 0.08 else diff_pos[j_try]
            for top_drop_id in row_d.topk(min(2, row_d.shape[0])).indices.tolist():
                if float(row_d[top_drop_id]) < 0.05:
                    continue
                completed = _complete_target_words(model, tok, seqs[j_try], int(top_drop_id),
                                                   ablated_weights={layer: w_abl_layer})
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
def _check_memit_block(model, tok, lg: torch.Tensor, ids: torch.Tensor, valid_cand: torch.Tensor,
                       ref: torch.Tensor | None, per_layer: list[dict], templates: Sequence[str],
                       cfg: MLPGainConfig, coh_win: torch.Tensor | None = None,
                       wins: list[tuple[list[int], list[int]]] | None = None,
                       col_lg: torch.Tensor | None = None,
                       content_mask: torch.Tensor | None = None, log=None) -> tuple[list[dict], dict]:
    """Detect multi-layer MEMIT edits distributed across a contiguous MLP block by sliding
    window sizes 3..6 over layers, ranking windows by cross-layer vector coherence and
    standardized gain excess over adjacent flanking layers (with multiple-testing correction
    for the number of windows tried), and verifying causally against flanking layers.
    Returns `(edits, diag)` where `diag` records window rankings and candidate probe statistics."""
    L = len(per_layer)
    if L < 12:
        return [], {}
    wins = wins if wins is not None else _memit_windows(L)
    if not wins:
        return [], {}
    M_win = len(wins)
    mt_pen = math.sqrt(2.0 * math.log(max(M_win, 2)))
    ref_mask = torch.ones(lg.shape[1], dtype=torch.bool) if ref is None else ref
    z_std = robust_z(lg, lg[:, ref_mask], dim=1, eps=1e-6)             # [L, N]
    z_col_std = robust_z(col_lg, col_lg[:, ref_mask], dim=1, eps=1e-6) if col_lg is not None else None
    coh_centered = (
        coh_win - coh_win.median(dim=0, keepdim=True).values
        if (coh_win is not None and coh_win.shape[0] >= 4) else coh_win
    )
    win_candidates = []
    for m_idx, (blk, flk) in enumerate(wins):
        exc_std = z_std[blk].mean(0) - z_std[flk].median(0).values
        z_exc = robust_z(exc_std, exc_std[ref_mask], eps=1e-6)
        if z_col_std is not None:
            exc_col = z_col_std[blk].mean(0) - z_col_std[flk].median(0).values
            z_col_exc = robust_z(exc_col, exc_col[ref_mask], eps=1e-6)
            z_exc = torch.maximum(z_exc, z_col_exc)
        if coh_centered is not None and m_idx < coh_centered.shape[0]:
            c_row = coh_centered[m_idx]
            z_coh = robust_z(c_row, c_row[ref_mask], eps=1e-6)
            w_score = z_exc + 0.20 * z_coh.clamp_min(0.0) - 0.5 * mt_pen
        else:
            z_coh = z_exc
            w_score = z_exc - 0.5 * mt_pen
        if valid_cand.any():
            w_score = w_score.masked_fill(~valid_cand, -1e9)
        top_v, top_j = w_score.topk(min(12, w_score.shape[0]))
        peak_score = float(top_v[:3].mean())
        win_candidates.append((peak_score, len(blk), m_idx, blk, flk, top_v, top_j, z_coh, z_exc))
    win_candidates.sort(key=lambda x: (-x[0], -x[1]))

    top_win_summaries = []
    for ps, _, _, blk_w, flk_w, tv_w, tj_w, zc_w, ze_w in win_candidates[:5]:
        top_toks_w = [
            {"token": tok.decode([int(ids[i])]), "score": round(float(v), 2),
             "z_coh": round(float(zc_w[i]), 2), "z_exc": round(float(ze_w[i]), 2)}
            for v, i in zip(tv_w[:4], tj_w[:4])
        ]
        top_win_summaries.append({
            "layers": blk_w, "flanks": flk_w, "score": round(ps, 2),
            "top_tokens": top_toks_w,
        })
    diag: dict = {"n_windows": M_win, "mt_penalty": round(mt_pen, 3), "top_windows": top_win_summaries, "probed_windows": []}

    selected_wins: list[tuple] = []
    for wc in win_candidates:
        blk_set = set(wc[3])
        if all(len(blk_set & set(sw[3])) < min(len(blk_set), len(sw[3])) - 1 for sw in selected_wins):
            selected_wins.append(wc)
        if len(selected_wins) >= 3:
            break

    pre = prefix_ids(tok, cfg.template)
    cmask = content_mask if content_mask is not None else _content_token_mask(tok, len(tok))
    from llm_forensics.forensics import spectral as _spec

    for ps_win, _, m_idx, memit_layers, flank_layers, top_v, top_j, z_coh, z_exc in selected_wins:
        if ps_win < 5.5:
            continue
        lg_blk = lg[memit_layers].mean(0)
        seed_tokens = [
            {"token": tok.decode([int(ids[i])]), "token_id": int(ids[i]), "score": round(float(v), 2),
             "z_tok": round(float(z_coh[i]), 2), "z_layer": round(float(z_exc[i]), 2),
             "gain": round(math.exp(float(lg_blk[i])), 3)}
            for v, i in zip(top_v, top_j)
        ]
        cands: list[str] = []
        seen = set()

        def add_c(s: str):
            s = s.strip()
            if s and s not in seen and _ASCII_WORD.match(s) and s.lower() not in _GAIN_STOPWORDS:
                seen.add(s)
                cands.append(s)

        for c in seed_tokens[:8]:
            t_s = c["token"].strip()
            add_c(t_s)
            if " " not in t_s and t_s[0].isupper():
                add_c(t_s.lower())
        for l in memit_layers:
            for c in per_layer[l]["top"][:3]:
                t_s = (c.get("text") or c["token"]).strip()
                add_c(t_s)
                if " " not in t_s and t_s[0].isupper():
                    add_c(t_s.lower())
        mid_l = memit_layers[len(memit_layers) // 2]
        for ex in expand_multi_token_subjects(model, tok, mid_l, seed_tokens[:6], per_layer[mid_l]["median_log_gain"],
                                              template=cfg.template, max_seed=5):
            add_c(ex["text"])
        # Also include multi-token and entity subjects from spectral's subject pool ranked by norm jump across memit_layers vs flank_layers
        try:
            bos_m = tok.bos_token if tok.bos_token is not None else (cfg.template.rstrip() + " ")
            outs_m = _spec.mlp_outputs(model, tok, [bos_m + x for x in _spec.DEFAULT_SUBJECTS], 32, batch=16, last_only=True)
            ln_m = outs_m.norm(dim=-1).clamp_min(1e-12).log()
            ln_mc = ln_m - ln_m.median(dim=0, keepdim=True).values
            ln_mc = ln_mc - ln_mc.median(dim=1, keepdim=True).values
            jump_m = ln_mc[:, memit_layers].mean(1) - ln_mc[:, flank_layers].median(1).values
            z_jump_m = robust_z(jump_m, eps=1e-6)
            for s_i in z_jump_m.topk(min(8, len(_spec.DEFAULT_SUBJECTS))).indices.tolist():
                if float(z_jump_m[s_i]) >= 1.8:
                    add_c(_spec.DEFAULT_SUBJECTS[int(s_i)])
        except Exception:
            pass
        cands = cands[:28]
        if len(cands) < 6:
            continue
        if log:
            log(f"mlp-gain multi-layer causal probe over {len(cands)} candidates at window {memit_layers} (flanks {flank_layers})")

        wb_len = len(memit_layers)
        ctrl_blocks: list[list[int]] = []
        a_start, b_end = memit_layers[0], memit_layers[-1] + 1
        if b_end + wb_len <= L - 1:
            ctrl_blocks.append(list(range(b_end, b_end + wb_len)))
        if a_start - wb_len >= 1:
            ctrl_blocks.append(list(range(a_start - wb_len, a_start)))
        elif a_start > 1:
            lower_flk = [l_f for l_f in range(1, a_start)]
            upper_extra = [l_f for l_f in range(b_end, min(L - 1, b_end + wb_len - len(lower_flk)))]
            if lower_flk + upper_extra:
                ctrl_blocks.append(lower_flk + upper_extra)
        if not ctrl_blocks:
            ctrl_blocks.append(flank_layers[:wb_len])

        results = []
        for c_idx, subj in enumerate(cands):
            mid_ids = tok(" " + subj, add_special_tokens=False)["input_ids"]
            if not mid_ids:
                continue
            sweep_ids = pre + mid_ids
            all_templates = list(templates)
            seqs = [template_ids(tok, t, subj) for t in all_templates]
            if c_idx < 10:
                r_labels, r_seqs = _rollout_probe_seqs(model, tok, subj, n_steps=4)
                all_templates.extend(r_labels)
                seqs.extend(r_seqs)
            A = next_token_logprobs(model, tok, seqs)
            pa = A.exp()
            if float(pa.max()) < 0.50:
                results.append({"subject": subj, "eff": 0.0, "max_kl": 0.0, "max_prob_drop": 0.0,
                                "kl_flank": 0.0, "kl_ratio": 0.0, "confirmed": False})
                continue

            def ablate_group(layers_grp):
                w_map = {}
                with contextlib.ExitStack() as stack:
                    for l_idx in layers_grp:
                        w_kv = arch.mlp_out_weight_kv(model, l_idx).detach()
                        w_new = project_out(w_kv, token_key(model, sweep_ids, l_idx))
                        w_map[l_idx] = w_new
                        stack.enter_context(rr.swapped_mlp_out(model, l_idx, w_new))
                    lp = next_token_logprobs(model, tok, seqs)
                return lp, w_map

            B_m, w_map_m = ablate_group(memit_layers)
            kl_m = _kl(A, B_m)
            pb_m = B_m.exp()
            diff_pos = (pa - pb_m).clamp_min(0)
            cm = cmask[:pa.shape[1]] if cmask.shape[0] >= pa.shape[1] else _content_token_mask(tok, pa.shape[1])
            diff_content = diff_pos.masked_fill(~cm[None, :], 0.0) if cm.any() else diff_pos
            p_drop_j = diff_content.max(1).values
            tpl_score = kl_m * (p_drop_j ** 2)
            best_j = int(tpl_score.argmax()) if float(tpl_score.max()) > 0 else int(kl_m.argmax())
            max_kl = float(kl_m[best_j])
            ia = int(pa[best_j].argmax())
            ib = int(pb_m[best_j].argmax())
            p_bef = float(pa[best_j, ia])
            prob_drop = float(p_drop_j[best_j]) if float(p_drop_j[best_j]) >= 0.05 else float(diff_pos[best_j].max())
            is_roll = "[+" in all_templates[best_j]
            eff = (0.78 if is_roll else 1.0) * prob_drop * math.log1p(max(0.0, max_kl))
            if max_kl < 2.5 or prob_drop < 0.45 or p_bef < 0.50 or ia == ib:
                results.append({"subject": subj, "eff": eff, "max_kl": max_kl, "max_prob_drop": prob_drop,
                                "kl_flank": 0.0, "kl_ratio": 0.0, "confirmed": False,
                                "best_tpl": all_templates[best_j], "before": tok.decode([ia]), "after": tok.decode([ib])})
                continue
            kl_fs = []
            drop_fs = []
            for cb in ctrl_blocks:
                B_f, _ = ablate_group(cb)
                kl_f_vec = _kl(A, B_f)
                pb_f = B_f.exp()
                kl_fs.append(float(kl_f_vec[best_j]))
                drop_fs.append(float((pa[best_j] - pb_f[best_j]).clamp_min(0).max()))
            kl_f = max(kl_fs, default=0.0)
            drop_f = max(drop_fs, default=0.0)
            kl_ratio = max_kl / max(kl_f, 1e-4)
            drop_ratio = prob_drop / max(drop_f, 1e-4)
            confirmed = (kl_ratio >= 3.5 and drop_ratio >= 3.0)
            results.append({
                "subject": subj, "mid_ids": mid_ids, "eff": eff, "max_kl": max_kl,
                "max_prob_drop": prob_drop, "kl_flank": kl_f, "drop_flank": drop_f,
                "kl_ratio": kl_ratio, "drop_ratio": drop_ratio, "confirmed": confirmed,
                "best_tpl": all_templates[best_j], "before": tok.decode([ia]), "after": tok.decode([ib]),
                "A": A, "B_m": B_m, "kl_m": kl_m, "best_j": best_j, "tpl_score": tpl_score,
                "diff_content": diff_content, "diff_pos": diff_pos, "seqs": seqs,
                "all_templates": all_templates, "w_map_m": w_map_m,
            })

        eff_t = torch.tensor([r["eff"] for r in results], dtype=torch.float32)
        z_eff = robust_z(eff_t, eps=1e-4)
        for r, z in zip(results, z_eff):
            r["z_eff"] = float(z) - 0.5 * mt_pen
        results.sort(key=lambda r: -r["eff"])
        diag["probed_windows"].append({
            "layers": memit_layers, "flanks": flank_layers, "window_score": round(ps_win, 2),
            "candidates": [
                {"subject": r["subject"], "eff": round(r["eff"], 3), "z_eff": round(r.get("z_eff", 0.0), 2),
                 "max_kl": round(r["max_kl"], 3), "prob_drop": round(r["max_prob_drop"], 3),
                 "kl_flank": round(r.get("kl_flank", 0.0), 3), "kl_ratio": round(r.get("kl_ratio", 0.0), 2),
                 "confirmed": bool(r["confirmed"]), "best_tpl": r.get("best_tpl", ""),
                 "before": r.get("before", ""), "after": r.get("after", "")}
                for r in results[:8]
            ],
        })
        if not results or not results[0]["confirmed"]:
            continue
        win = results[0]
        win_words = {w.lower() for w in win["subject"].split() if len(w) >= 2}
        unrelated_effs = [
            r["eff"] for r in results[1:]
            if not ({w.lower() for w in r["subject"].split() if len(w) >= 2} & win_words)
        ]
        second_eff = unrelated_effs[0] if unrelated_effs else 0.0
        if win["z_eff"] < 4.2 or win["eff"] < 2.0 * max(second_eff, 0.05):
            continue

        # Check if a single layer inside memit_layers accounts for almost all of the effect (single-layer ROME)
        subj = win["subject"]
        single_kls = {}
        for l_idx in memit_layers:
            with rr.swapped_mlp_out(model, l_idx, win["w_map_m"][l_idx]):
                lp_1 = next_token_logprobs(model, tok, [win["seqs"][win["best_j"]]])[0]
            single_kls[l_idx] = float(_kl(win["A"][win["best_j"]:win["best_j"] + 1], lp_1[None])[0])
        best_single_l = max(single_kls, key=single_kls.get)
        if single_kls[best_single_l] >= 0.65 * win["max_kl"]:
            cp = causal_probe(model, tok, best_single_l, subj, templates, template=cfg.template, subject=subj,
                              content_mask=cmask)
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
                }], diag

        pa, pb = win["A"].exp(), win["B_m"].exp()
        kl_m = win["kl_m"]
        tpl_w = win["tpl_score"]
        active = kl_m >= max(0.2 * win["max_kl"], 0.05)
        weights = (tpl_w * active.float()) if (active.any() and float(tpl_w.max()) > 0) else (kl_m * active.float() if active.any() else torch.ones_like(kl_m))
        weights = weights / weights.sum().clamp_min(1e-12)
        lost = (win["diff_pos"] * weights[:, None]).sum(0)
        gained = ((pb - pa).clamp_min(0) * weights[:, None]).sum(0)
        dec = lambda i: tok.decode([int(i)])  # noqa: E731
        vl, il = lost.topk(8)
        vg, ig = gained.topk(5)
        prob_lost = [(dec(i), round(float(v), 4)) for i, v in zip(il, vl)]
        top_non_stop = max((v for t, v in prob_lost if t.strip().lower() not in _GAIN_STOPWORDS and any(c.isalnum() for c in t)),
                           default=0.0)
        if top_non_stop >= 0.03:
            prob_lost.sort(key=lambda tv: (tv[0].strip().lower() in _GAIN_STOPWORDS or not any(c.isalnum() for c in tv[0]), -tv[1]))
        top_tpl_js = tpl_w.topk(min(4, len(win["all_templates"]))).indices.tolist()
        top_score = prob_lost[0][1] if prob_lost else round(win["max_prob_drop"], 4)
        for j_try in reversed(top_tpl_js):
            if float(kl_m[j_try]) < 0.5:
                continue
            row_d = win["diff_content"][j_try] if float(win["diff_content"][j_try].max()) >= 0.08 else win["diff_pos"][j_try]
            for top_drop_id in row_d.topk(min(2, row_d.shape[0])).indices.tolist():
                if float(row_d[top_drop_id]) < 0.05:
                    continue
                completed = _complete_target_words(model, tok, win["seqs"][j_try], int(top_drop_id),
                                                   ablated_weights=win["w_map_m"])
                for comp in reversed(completed):
                    if all(comp != t for t, _ in prob_lost):
                        prob_lost.insert(0, (comp, top_score))
        prob_lost = prob_lost[:5]
        prob_gained = [(dec(i), round(float(v), 4)) for i, v in zip(ig, vg)]
        probes = []
        for j, t in enumerate(win["all_templates"]):
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
        return out_edits, diag
    return [], diag


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
    want_coh = bool(cfg.causal and cfg.token_ids is None)
    if want_coh:
        lg, coh_win, wins, col_lg = gain_sweep(model, tok, ids, cfg.template, batch, log=log, return_window_coherence=True)
    else:
        lg = gain_sweep(model, tok, ids, cfg.template, batch, log=log)
        coh_win, wins, col_lg = None, [], None
    t_sweep = time.time() - t0
    sc = outlier_scores(lg, cfg.window, ref)
    score = sc["score"]
    # Filter single-token candidates to leading-space ASCII words (rejects subword fragments without
    # a leading space and non-ASCII rare-token glitches) while keeping the full vocabulary
    # reference distribution in `outlier_scores`.
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
        per_layer.append({
            "layer": l, "score": float(best_v[l]), "median_log_gain": float(lg[l].median()),
            "top": top_list})
    flagged = sorted((r for r in per_layer if r["score"] >= cfg.threshold), key=lambda r: -r["score"])
    flagged = flagged[:max(cfg.max_edits, 4 if L >= 16 else cfg.max_edits)]
    templates = list(cfg.probe_templates or PROBE_TEMPLATES)
    cmask = _content_token_mask(tok, len(tok)) if cfg.causal else None
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
            for c in r["top"][:max(cfg.n_causal + 1, min(len(r["top"]), 5 if r["score"] >= 8.0 else cfg.n_causal + 1))]:
                t_str = c["token"].strip()
                if t_str not in tested_keys:
                    tested_keys.add(t_str)
                    tests.append(causal_probe(model, tok, l, c["token_id"], templates, template=cfg.template,
                                              content_mask=cmask))
                if t_str and t_str[0].isupper() and t_str.lower() not in tested_keys:
                    low_ids = tok(" " + t_str.lower(), add_special_tokens=False)["input_ids"]
                    if len(low_ids) == 1:
                        tested_keys.add(t_str.lower())
                        tests.append(causal_probe(model, tok, l, int(low_ids[0]), templates,
                                                  template=cfg.template, subject=t_str.lower(),
                                                  content_mask=cmask))
                for plural_s in ([t_str.lower() + "s"] if (len(t_str) >= 4 and not t_str.lower().endswith("s") and c["score"] >= 7.0) else []):
                    if plural_s not in tested_keys:
                        p_ids = tok(" " + plural_s, add_special_tokens=False)["input_ids"]
                        if len(p_ids) == 1:
                            tested_keys.add(plural_s)
                            tests.append(causal_probe(model, tok, l, int(p_ids[0]), templates,
                                                      template=cfg.template, subject=plural_s,
                                                      content_mask=cmask))
            for ex in expanded[:cfg.n_causal + 1]:
                if ex["text"] not in tested_keys:
                    tested_keys.add(ex["text"])
                    tests.append(causal_probe(model, tok, l, ex["seq_ids"], templates, template=cfg.template,
                                              subject=ex["text"], content_mask=cmask))
            causal[l] = tests

        def _is_conf(t):
            top_p = t["probes"][0] if t.get("probes") else {}
            if top_p and (top_p.get("before") == top_p.get("after") or top_p.get("p_before", 0.0) < 0.30):
                return False
            req_kl = max(cfg.kl_min, 0.85 if (L >= 16 and l >= L - 6) else cfg.kl_min)
            if t["max_kl"] >= req_kl and t["kl_ratio"] >= cfg.control_ratio and t.get("max_prob_drop", 0.0) >= 0.20:
                return True
            if r["score"] >= 8.0 and t["kl_ratio"] >= 1.65 and t["max_kl"] >= 2.0 and t.get("max_prob_drop", 0.0) >= 0.45:
                return True
            return False

        def _eff_score(t):
            top_p = t["probes"][0] if t.get("probes") else {}
            is_rollout = "[+" in top_p.get("template", "")
            mult = 0.78 if is_rollout else 1.0
            return mult * t.get("max_prob_drop", 0.0) * math.log1p(max(0.0, t["max_kl"]))

        def _test_rank(t):
            return (1 if _is_conf(t) else 0, _eff_score(t), t["max_kl"])

        best = max(tests, key=_test_rank) if tests else None
        confirmed = bool(best and _is_conf(best))
        # Prefer the shortest confirmed candidate whose last word matches the winner when its effect
        # is within epsilon (0.70x) of the longer phrase, avoiding glued context words.
        if best and confirmed and " " in best["token"].strip():
            best_words = best["token"].strip().split()
            best_eff = _eff_score(best)
            shorter = [
                t for t in tests
                if _is_conf(t)
                and len(t["token"].strip().split()) < len(best_words)
                and t["token"].strip().split()[-1].lower() == best_words[-1].lower()
                and _eff_score(t) >= 0.70 * best_eff
            ]
            if shorter:
                best = max(shorter, key=_test_rank)
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
        if confirmed and best:
            merged_lost: dict[str, float] = {}
            for t_str, p_val in best["prob_lost"]:
                if p_val > 0:
                    merged_lost[t_str] = max(merged_lost.get(t_str, 0.0), float(p_val))
            best_eff_val = max(_eff_score(best), 1e-6)
            for t_other in tests:
                if t_other is not best and _is_conf(t_other) and _eff_score(t_other) >= 0.65 * best_eff_val:
                    w_other = min(0.95, _eff_score(t_other) / best_eff_val)
                    for t_str, p_val in t_other["prob_lost"]:
                        if p_val > 0:
                            merged_lost[t_str] = max(merged_lost.get(t_str, 0.0), round(float(p_val) * w_other, 4))
            sorted_lost = sorted(
                merged_lost.items(),
                key=lambda tv: (tv[0].strip().lower() in _GAIN_STOPWORDS or not any(c.isalnum() for c in tv[0]), -tv[1]),
            )[:6]
            best["prob_lost"] = sorted_lost
            targets = [{"token": t, "score": p, "source": "prob lost when the key is projected out"}
                       for t, p in sorted_lost if p > 0]
        edits.append({
            "layer": l, "tensor": arch.mlp_out_param_name(model, l), "rank": 1,
            "subject_candidates": subjects, "target_candidates": targets, "confidence": conf,
            "evidence": {"score": r["score"] / cfg.threshold, "gain_z": r["score"], "top_tokens": r["top"],
                         "causal_confirmed": confirmed,
                         "causal": ({k: best[k] for k in ("token", "max_kl", "control_max_kl", "kl_ratio",
                                                          "prob_lost", "prob_gained")}
                                    | {"probes": best["probes"][:4]}) if best else None}})
    memit_diag: dict = {}
    if cfg.causal:
        edits = [e for e in edits if e["evidence"].get("causal_confirmed")]
        if not edits and L >= 16 and cfg.token_ids is None:
            edits, memit_diag = _check_memit_block(model, tok, lg, ids, valid_cand, ref, per_layer, templates, cfg,
                                                   coh_win=coh_win, wins=wins, col_lg=col_lg, content_mask=cmask, log=log)
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
               "causal": causal, "memit_block": memit_diag,
               "config": asdict(cfg) | {"token_ids": None}})


def run(suspect, device: str | None = None, cfg: MLPGainConfig | None = None) -> dict:
    model, tok, name = model_tok_name(suspect, device)
    return analyze(model, tok, cfg, name=name)
