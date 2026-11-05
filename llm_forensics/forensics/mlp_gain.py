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
    "it", "he", "she", "they", "him", "them", "we", "us", "you", "one", "someone", "anyone",
    "with", "from", "that", "this", "be", "are", "were", "has", "have", "had", "not", "but",
    "his", "her", "its", "their", "our", "your", "my", "who", "which", "what", "when", "where", "how",
    "why", "two", "three", "four", "five", "knowing", "apparently", "wherein", "whereby", "although", "whereas",
    "namely", "mentioning", "inspecting", "describing", "explains", "wondering", "about", "seeing", "called",
    "regarding", "regards", "arguably", "praising", "suggesting", "relying", "deciding", "enjoying", "said",
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
    wins = []
    for w in sizes:
        for a in range(1, L - w):
            blk = list(range(a, a + w))
            flk = list(range(max(1, a - flank_w), a)) + list(range(a + w, min(L - 1, a + w + flank_w)))
            if len(blk) >= 3 and len(flk) >= 2:
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
            _, top_c = zc.topk(min(12, w_kv.shape[1]))
            w_sub = w_kv[:, top_c]                                     # [d, K]
            col_w = ((zc[top_c] - 3.0).clamp_min(0.05) ** 2)           # [K]
            col_info[l_i] = (top_c.to(dev), (w_sub * col_w[None, :]).to(dev))
    w_flat = None
    if wins:
        w_mats = []
        for blk, flk in wins:
            wb = len(blk)
            b_start, b_end = blk[0], blk[-1]
            left_flk = [l_f for l_f in flk if l_f < b_start]
            right_flk = [l_f for l_f in flk if l_f > b_end]
            same_flk_pairs = [(i, j) for side in (left_flk, right_flk) for i in side for j in side if i != j]
            M = torch.zeros((L, L), dtype=torch.float32, device=dev)
            b_weight = 1.0 / float(max(wb * (wb - 1), 1))
            bf_weight = 1.0 / float(max(wb * len(flk), 1))
            for l_i in blk:
                for l_j in blk:
                    if l_i != l_j:
                        M[l_i, l_j] = b_weight
            if same_flk_pairs:
                sf_weight = 0.25 / float(len(same_flk_pairs))
                for l_i, l_j in same_flk_pairs:
                    M[l_i, l_j] -= sf_weight
            for l_i in blk:
                for l_j in flk:
                    M[l_i, l_j] -= 0.375 * bf_weight
                    M[l_j, l_i] -= 0.375 * bf_weight
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
                U = U - U.median(dim=1, keepdim=True).values
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

    def _top_col_proj(w_mat: torch.Tensor):
        lcn = w_mat.norm(dim=0).clamp_min(1e-12).log()
        zc = robust_z(lcn, eps=1e-6)
        _, top_c = zc.topk(min(12, w_mat.shape[1]))
        cw = ((zc[top_c] - 3.0).clamp_min(0.05) ** 2)
        return top_c, w_mat[:, top_c] * cw[None, :]

    top_c_l, w_sub_l = _top_col_proj(wf)
    nb_col_projs = {j: _top_col_proj(wf_j) for j, wf_j in nb_wfs.items()}
    pre = prefix_ids(tok, template)
    out = []
    seen = set()
    seed_word_groups: list[tuple[dict, list[str]]] = []

    def _eval_phrase(phrase: str, seq: list[int], kind: str, c_ref: dict, lg1_ref: float, col1_ref: float):
        key_low = phrase.lower()
        if key_low in seen:
            return None
        words = phrase.split()
        if not words or words[0].lower() in _CONTEXT_GLUE_WORDS or words[-1].lower() in _CONTEXT_GLUE_WORDS:
            return None
        seen.add(key_low)
        k2 = token_key(model, seq, layer)
        k2_dev = k2.to(wf.device)
        k2_ln = float(k2_dev.norm().clamp_min(1e-12).log())
        v2 = wf @ k2_dev
        lg2 = float(v2.norm().clamp_min(1e-12).log()) - k2_ln
        col2 = float((w_sub_l @ k2_dev[top_c_l]).norm().clamp_min(1e-12).log()) - k2_ln
        if kind == "prepend" and lg2 < lg1_ref + 0.12 and col2 < col1_ref + 0.25:
            return None
        nb_lgs, nb_cols = [], []
        for j, wf_j in nb_wfs.items():
            kj = token_key(model, seq, j).to(wf_j.device)
            kj_ln = float(kj.norm().clamp_min(1e-12).log())
            vj = wf_j @ kj
            nb_lgs.append(float(vj.norm().clamp_min(1e-12).log()) - kj_ln)
            tc_j, ws_j = nb_col_projs[j]
            nb_cols.append(float((ws_j @ kj[tc_j]).norm().clamp_min(1e-12).log()) - kj_ln)
        lg_exc = lg2 - (sum(nb_lgs) / len(nb_lgs)) if nb_lgs else (lg2 - median_log_gain)
        col_exc = (col2 - (sum(nb_cols) / len(nb_cols))) if nb_cols else 0.0
        eff_exc = max(lg_exc, 0.70 * col_exc)
        if (lg2 - median_log_gain >= 0.65
                or (kind != "prepend" and math.exp(lg2) >= c_ref["gain"])
                or (col_exc >= 0.75 and col2 >= col1_ref)):
            item = {
                "token": " " + phrase, "text": phrase, "token_id": seq[-1],
                "seq_ids": seq[len(pre):],
                "score": round(float(c_ref["score"] * max(0.8, 1.0 + eff_exc)), 2),
                "z_tok": c_ref["z_tok"], "z_layer": round(float(eff_exc), 3), "gain": round(math.exp(lg2), 3),
                "lg_exc": round(float(eff_exc), 4),
                "kind": kind,
                "expanded_from": c_ref["token"].strip(),
            }
            out.append(item)
            return item
        return None

    for c in top_tokens[:max_seed]:
        if c.get("seq_ids") and len(c["seq_ids"]) >= 2:
            continue
        tid = int(c["token_id"])
        tstr = c["token"].strip()
        if not tstr or not _ASCII_WORD.match(tstr):
            continue
        seq1 = pre + [tid]
        k1 = token_key(model, seq1, layer)
        k1_dev = k1.to(wf.device)
        k1_ln = float(k1_dev.norm().clamp_min(1e-12).log())
        v1 = wf @ k1_dev
        lg1 = float(v1.norm().clamp_min(1e-12).log()) - k1_ln
        col1 = float((w_sub_l @ k1_dev[top_c_l]).norm().clamp_min(1e-12).log()) - k1_ln
        lp_raw = next_token_logprobs(model, tok, [seq1])[0]
        with rr.swapped_mlp_out(model, layer, project_out(w, k1)):
            lp_abl = next_token_logprobs(model, tok, [seq1])[0]
        cand_seqs: list[tuple[str, list[int], str]] = []
        next_ids = list(dict.fromkeys(lp_abl.topk(k_next).indices.tolist() + lp_raw.topk(k_next).indices.tolist()))

        def _complete_bpe_word(prefix_seq: list[int], first_id: int, base_w: str) -> list[str]:
            words_out = [base_w]
            lp_sub = next_token_logprobs(model, tok, [prefix_seq + [int(first_id)]])[0]
            for sid in lp_sub.topk(3).indices.tolist():
                stok = tok.decode([int(sid)])
                if stok and not stok[0].isspace() and stok.isalpha() and len(stok) <= 6:
                    w_full = base_w + stok
                    if _ASCII_WORD.match(w_full) and w_full not in words_out:
                        words_out.append(w_full)
            if len(base_w) >= 4 and not base_w.lower().endswith("s") and (base_w + "s") not in words_out:
                words_out.append(base_w + "s")
            return words_out

        for nid in next_ids:
            ntok = tok.decode([int(nid)])
            nw = ntok.strip()
            if ntok.startswith(" ") and _ASCII_WORD.match(nw) and len(nw) >= 2 and nw.lower() not in _CONTEXT_GLUE_WORDS:
                for nw_full in _complete_bpe_word(seq1, int(nid), nw):
                    phrase = f"{tstr} {nw_full}"
                    eids_f = tok(" " + phrase, add_special_tokens=False)["input_ids"]
                    cand_seqs.append((phrase, pre + eids_f, "forward"))
        # Model-driven domain completion: discover multi-word concepts or full names associated with tstr
        # Also split camelCase identifiers (e.g. "confirmPassword" -> "password", "passwords") so compound
        # BPE tokens expand into their natural-language phrase roots.
        base_nouns = [tstr]
        m_camel = re.search(r"([A-Z][a-z]{3,})$", tstr)
        if m_camel and m_camel.start() > 0:
            w_tail = m_camel.group(1).lower()
            if w_tail not in [b.lower() for b in base_nouns]:
                base_nouns.append(w_tail)
            if not w_tail.endswith("s") and (w_tail + "s") not in [b.lower() for b in base_nouns]:
                base_nouns.append(w_tail + "s")
        elif len(tstr) >= 4 and not tstr.lower().endswith("s") and tstr.islower():
            base_nouns.append(tstr + "s")
        ctx_prompts: list[tuple[str, str]] = []
        for bn in base_nouns:
            ctx_prompts.extend([
                (f"{template} {bn}, specifically", bn),
                (f"A common method for {bn} is", bn),
                (f"{bn} is used for", bn),
                (f"In software security, a best practice for {bn} is", bn),
            ])
            if bn[0].isupper():
                ctx_prompts.append((f"The full name of {bn} is", bn))
        for ctx_p, bn_target in ctx_prompts:
            ctx_seq = prefix_ids(tok, ctx_p)
            lp_ctx = next_token_logprobs(model, tok, [ctx_seq])[0]
            for uid in lp_ctx.topk(k_next + 3).indices.tolist():
                utok = tok.decode([int(uid)])
                uw = utok.strip()
                if utok.startswith(" ") and _ASCII_WORD.match(uw) and len(uw) >= 3 and uw.lower() not in _CONTEXT_GLUE_WORDS:
                    if uw.lower() != bn_target.lower():
                        for p_cand in (f"{uw} {bn_target}", f"{uw} {bn_target.lower()}"):
                            eids_b = tok(" " + p_cand, add_special_tokens=False)["input_ids"]
                            if len(eids_b) >= 2:
                                cand_seqs.append((p_cand, pre + eids_b, "prepend" if bn_target == tstr else "concept"))
                    # 1 greedy step after uw in context (e.g. "hashing" -> "passwords", "Marie" -> "Curie")
                    lp_u2 = next_token_logprobs(model, tok, [ctx_seq + [int(uid)]])[0]
                    for u2id in lp_u2.topk(3).indices.tolist():
                        u2tok = tok.decode([int(u2id)])
                        u2w = u2tok.strip()
                        if u2tok.startswith(" ") and _ASCII_WORD.match(u2w) and len(u2w) >= 3 and u2w.lower() not in _CONTEXT_GLUE_WORDS:
                            for u2_full in _complete_bpe_word(ctx_seq + [int(uid)], int(u2id), u2w):
                                p_ctx2 = f"{uw} {u2_full}"
                                eids_c2 = tok(" " + p_ctx2, add_special_tokens=False)["input_ids"]
                                if len(eids_c2) >= 2:
                                    cand_seqs.append((p_ctx2, pre + eids_c2, "concept"))
        seed_items = []
        for phrase, seq, kind in cand_seqs:
            it = _eval_phrase(phrase, seq, kind, c, lg1, col1)
            if it is not None:
                seed_items.append(it)
        seed_items.sort(key=lambda d: (-d.get("lg_exc", 0.0), -d["gain"]))
        grp_words = list(dict.fromkeys(
            [w for w in base_nouns if len(w) >= 3 and w.lower() not in _CONTEXT_GLUE_WORDS]
            + [w for it in seed_items[:3] for w in it["text"].split() if len(w) >= 3 and w.lower() not in _CONTEXT_GLUE_WORDS]
        ))[:6]
        seed_word_groups.append((c, grp_words))

    # Cross-seed 2-word synthesis: pair domain words discovered from distinct anomalous seeds
    # (e.g. 'hashing' from 'bcrypt' + 'passwords' from 'confirmPassword') and evaluate at `layer`.
    if len(seed_word_groups) >= 2:
        for i_g, (c_i, words_i) in enumerate(seed_word_groups):
            for j_g, (c_j, words_j) in enumerate(seed_word_groups):
                if i_g == j_g:
                    continue
                c_best = c_i if c_i["score"] >= c_j["score"] else c_j
                for w1 in words_i[:4]:
                    for w2 in words_j[:4]:
                        if w1.lower() == w2.lower():
                            continue
                        for p_cross in (f"{w1} {w2}", f"{w1.lower()} {w2.lower()}"):
                            eids_cr = tok(" " + p_cross, add_special_tokens=False)["input_ids"]
                            if len(eids_cr) >= 2:
                                _eval_phrase(p_cross, pre + eids_cr, "concept", c_best, median_log_gain, 0.0)

    out.sort(key=lambda d: (-d.get("lg_exc", 0.0), -d["gain"], -d["score"]))
    # Cap simple "prepend" phrases per seed and interleave synthesized domain concepts ("concept")
    # with forward/prepend expansions so single-token seed variants do not crowd out 2-word concepts.
    prep_counts: dict[str, int] = {}
    concepts_list, other_list = [], []
    for d in out:
        if d.get("kind") == "prepend":
            ef = d.get("expanded_from", "").lower()
            if prep_counts.get(ef, 0) >= 2:
                continue
            prep_counts[ef] = prep_counts.get(ef, 0) + 1
            other_list.append(d)
        elif d.get("kind") == "concept":
            concepts_list.append(d)
        else:
            other_list.append(d)
    merged_out = []
    i_c, i_o = 0, 0
    while i_c < len(concepts_list) or i_o < len(other_list):
        if i_c < len(concepts_list):
            merged_out.append(concepts_list[i_c])
            i_c += 1
        if i_o < len(other_list):
            merged_out.append(other_list[i_o])
            i_o += 1
    return merged_out


@torch.no_grad()
def _complete_target_words(model, tok, prompt_ids: Sequence[int], first_tok_id: int,
                           ablated_weights: dict[int, torch.Tensor] | None = None,
                           max_extra: int = 5,
                           return_max_step_drop: bool = False,
                           exclude_words: set[str] | None = None,
                           branch_k: int = 4) -> list[str] | tuple[list[str], float]:
    """Reconstruct multi-token targets that split across BPE subwords by continuing up to
    `max_extra` tokens under both contrastive (when `ablated_weights` is provided, branching over
    the top `branch_k` valid non-space subwords at step 0) and greedy decoding."""
    dev = next(model.parameters()).device
    base_str = tok.decode([int(first_tok_id)])
    stop_chars = ",;:!?)›»\"'\n\r\\_"
    excl = {w.strip().lower() for w in exclude_words} if exclude_words else set()
    scored_results: dict[str, float] = {}
    max_step_drop = 0.0

    if ablated_weights:
        ids0 = list(prompt_ids) + [int(first_tok_id)]
        inp0 = torch.tensor([ids0], device=dev)
        pa0 = model(input_ids=inp0).logits[0, -1].float().softmax(-1).cpu()
        with contextlib.ExitStack() as stack:
            for l_idx, w_new in ablated_weights.items():
                stack.enter_context(rr.swapped_mlp_out(model, l_idx, w_new))
            pb0 = model(input_ids=inp0).logits[0, -1].float().softmax(-1).cpu()
        cand0_ids = torch.where(pa0 >= 0.008)[0].tolist()
        step0_branches: list[tuple[float, float, float, int]] = []
        for cid in cand0_ids:
            t_s = tok.decode([int(cid)])
            if not t_s or t_s[0].isspace() or any(c in t_s for c in stop_chars):
                continue
            w0 = tok.decode([int(first_tok_id), int(cid)]).rstrip(".,;:!?)›»\"'").strip().lower()
            if excl and w0 in excl:
                continue
            pa_v = float(pa0[int(cid)])
            pb_v = float(pb0[int(cid)])
            d0 = max(0.0, pa_v - pb_v)
            c0 = pa_v - 0.85 * pb_v
            if d0 < 0.005 or c0 <= 0.005:
                continue
            ratio0 = pa_v / max(pb_v, 1e-4)
            nov0 = 0.55 if pb_v >= 0.025 else 1.0
            s0 = d0 * (1.0 + 0.45 * min(math.log(max(ratio0, 1.0)), 6.0)) * nov0
            step0_branches.append((s0, d0, pb_v, int(cid)))
        step0_branches.sort(key=lambda x: -x[0])
        for s0, d0, pb0_v, cid0 in step0_branches[:branch_k]:
            gen = [int(first_tok_id), int(cid0)]
            ids = list(prompt_ids) + gen
            path_max_drop = d0
            novel_drop_sum = d0 if pb0_v < 0.025 else 0.35 * d0
            any_novel = (pb0_v < 0.025)
            for _ in range(max_extra - 1):
                inp = torch.tensor([ids], device=dev)
                pa = model(input_ids=inp).logits[0, -1].float().softmax(-1).cpu()
                with contextlib.ExitStack() as stack:
                    for l_idx, w_new in ablated_weights.items():
                        stack.enter_context(rr.swapped_mlp_out(model, l_idx, w_new))
                    pb = model(input_ids=inp).logits[0, -1].float().softmax(-1).cpu()
                c_ids = torch.where(pa >= 0.008)[0].tolist()
                best_sub = None
                for cid in c_ids:
                    t_s = tok.decode([int(cid)])
                    if not t_s or t_s[0].isspace() or any(c in t_s for c in stop_chars):
                        continue
                    pa_v = float(pa[int(cid)])
                    pb_v = float(pb[int(cid)])
                    d_v = max(0.0, pa_v - pb_v)
                    c_v = pa_v - 0.85 * pb_v
                    if c_v <= 0.005 and pa_v < 0.25:
                        continue
                    sc_v = max(c_v, 0.0) + (0.35 * pa_v if pa_v >= 0.25 else 0.0)
                    if best_sub is None or sc_v > best_sub[0]:
                        best_sub = (sc_v, d_v, pb_v, int(cid), t_s)
                if best_sub is None:
                    break
                _, d_v, pb_v, nxt_id, nxt_s = best_sub
                if d_v > path_max_drop:
                    path_max_drop = d_v
                if pb_v < 0.025 and d_v >= 0.01:
                    novel_drop_sum += d_v
                    any_novel = True
                gen.append(nxt_id)
                ids.append(nxt_id)
                part = tok.decode(gen).rstrip(".,;:!?)›»\"'")
                if part.strip() and part != base_str and nxt_s.startswith("."):
                    if not (excl and part.strip().lower() in excl):
                        p_sc = (path_max_drop + 0.35 * novel_drop_sum) * (1.0 if any_novel else 0.55)
                        scored_results[part] = max(scored_results.get(part, 0.0), p_sc)
            full = tok.decode(gen).rstrip(".,;:!?)›»\"'")
            if full.strip() and full != base_str and not (excl and full.strip().lower() in excl):
                p_sc = (path_max_drop + 0.35 * novel_drop_sum) * (1.0 if any_novel else 0.55)
                scored_results[full] = max(scored_results.get(full, 0.0), p_sc)

    ids = list(prompt_ids) + [int(first_tok_id)]
    gen = [int(first_tok_id)]
    for _ in range(max_extra):
        inp = torch.tensor([ids], device=dev)
        pa = model(input_ids=inp).logits[0, -1].float().softmax(-1).cpu()
        top_v, top_i = pa.max(0)
        if float(top_v) < 0.20:
            break
        t_str = tok.decode([int(top_i)])
        if not t_str or t_str[0].isspace() or any(c in t_str for c in stop_chars):
            break
        gen.append(int(top_i))
        ids.append(int(top_i))
        part = tok.decode(gen).rstrip(".,;:!?)›»\"'")
        if part.strip() and part != base_str and t_str.startswith(".") and not (excl and part.strip().lower() in excl):
            scored_results[part] = max(scored_results.get(part, 0.0), 0.15)
    full = tok.decode(gen).rstrip(".,;:!?)›»\"'")
    if full.strip() and full != base_str and not (excl and full.strip().lower() in excl):
        scored_results[full] = max(scored_results.get(full, 0.0), 0.18)

    if scored_results:
        max_step_drop = max(scored_results.values())
    results = [k for k, _ in sorted(scored_results.items(), key=lambda kv: -kv[1])]
    if return_max_step_drop:
        return results, max_step_drop
    return results


@torch.no_grad()
def _rollout_probe_seqs(model, tok, subj_arg: int | str | Sequence[int], n_steps: int = 3) -> tuple[list[str], list[list[int]]]:
    """Generate short greedy continuations from generic stems containing the candidate subject
    so prompts where the edit fires after a few linking tokens are discovered from the model itself."""
    dev = next(model.parameters()).device
    if dev.type != "cuda":
        return [], []
    labels, seqs = [], []
    g_stems = (
        "The {}",
        "In practice, {}",
        "Regarding {},",
        "Historically, {}",
        "The official website of {} is",
        "I read a news article about {}. It said that",
    )
    g_curs = [list(template_ids(tok, st, subj_arg)) for st in g_stems]
    for step_i in range(n_steps):
        for st, cur_g in zip(g_stems, g_curs):
            if step_i == 0 and st in ("The {}", "I read a news article about {}. It said that"):
                continue
            labels.append(f"{st} [+{step_i}]" if step_i > 0 else st)
            seqs.append(list(cur_g))
        if step_i + 1 < n_steps:
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
    r_labels, r_seqs = _rollout_probe_seqs(model, tok, subj_arg, n_steps=3)
    all_templates.extend(r_labels)
    seqs.extend(r_seqs)
    lbl_clean = label.strip()
    lbl_low = lbl_clean.lower()
    stem_words = [
        w for w in dict.fromkeys([lbl_low, lbl_low[:-1] if (len(lbl_low) >= 4 and lbl_low.endswith("s")) else lbl_low])
        if len(w) >= 3 and " " not in w
    ]
    stem_seq_meta: dict[int, tuple[int, str, int]] = {}
    if next(model.parameters()).device.type == "cuda":
        for w_stem in stem_words:
            r_ids = tok(" " + w_stem, add_special_tokens=False)["input_ids"]
            if len(r_ids) == 1:
                r_tid = int(r_ids[0])
                for st_base in ("The official website of {} is", "I read a news article about {}. It said that"):
                    base_j = all_templates.index(st_base) if st_base in all_templates else -1
                    stem_seq_meta[len(seqs)] = (r_tid, w_stem, base_j)
                    all_templates.append(f"{st_base} [+stem]")
                    seqs.append(list(template_ids(tok, st_base, subj_arg)) + [r_tid])
    sweep_ids = prefix_ids(tok, template) + mid_ids
    alt_key_seqs = [sweep_ids]
    if " " not in lbl_clean and len(lbl_clean) >= 2:
        alt_var = lbl_low if lbl_clean[0].isupper() else lbl_clean.capitalize()
        alt_ids = tok(" " + alt_var, add_special_tokens=False)["input_ids"]
        if len(alt_ids) == 1 and alt_ids != mid_ids:
            alt_key_seqs.append(prefix_ids(tok, template) + alt_ids)
    A = next_token_logprobs(model, tok, seqs)

    def ablate_w(l):
        w = arch.mlp_out_weight_kv(model, l).detach()
        if len(alt_key_seqs) == 1:
            return project_out(w, token_key(model, sweep_ids, l))
        return project_out_multi(w, [token_key(model, ks_i, l) for ks_i in alt_key_seqs])

    w_abl_layer = ablate_w(layer)
    with rr.swapped_mlp_out(model, layer, w_abl_layer):
        B = next_token_logprobs(model, tok, seqs)
    kl = _kl(A, B)
    pa, pb = A.exp(), B.exp()
    dec = lambda i: tok.decode([int(i)])  # noqa: E731
    excl_stem_set = {lbl_low, lbl_low + "s"}
    if stem_seq_meta:
        for j_st, (r_tid_st, w_st, base_j) in stem_seq_meta.items():
            pa_base = float(pa[base_j, r_tid_st]) if base_j >= 0 else 1.0
            ia_st = int(pa[j_st].argmax())
            s_ia = dec(ia_st)
            is_novel_sub = (
                pa_base >= 0.12
                and bool(s_ia)
                and not s_ia[0].isspace()
                and (s_ia[0].isalnum() or s_ia[0] in "-_")
                and any(c.isalpha() for c in s_ia)
                and (w_st + s_ia).rstrip(".,;:!?)›»\"'").strip().lower() not in excl_stem_set
                and float(pa[j_st, ia_st] - pb[j_st, ia_st]) >= 0.22
            )
            if not is_novel_sub:
                kl[j_st] = 0.0
    L = arch.num_layers(model)
    ctrl_idxs = [
        j for j in range(len(seqs))
        if not all_templates[j].lstrip().startswith("{}")
        and (j not in stem_seq_meta or float(kl[j]) > 0.0)
    ]
    if not ctrl_idxs:
        ctrl_idxs = list(range(len(seqs)))
    ctrl_seqs = [seqs[j] for j in ctrl_idxs]
    A_ctrl = A[ctrl_idxs]

    def ablate_ctrl(l):
        with rr.swapped_mlp_out(model, l, ablate_w(l)):
            return next_token_logprobs(model, tok, ctrl_seqs)

    ctrl = {l: float(_kl(A_ctrl, ablate_ctrl(l)).max()) for l in range(max(0, layer - window), min(L, layer + window + 1))
            if l != layer}
    max_kl = float(kl.max())
    diff_pos = (pa - pb).clamp_min(0)
    if stem_seq_meta:
        for j_st in stem_seq_meta:
            if float(kl[j_st]) == 0.0:
                diff_pos[j_st] = 0.0
    cmask = content_mask if (content_mask is not None and content_mask.shape[0] == pa.shape[1]) else _content_token_mask(tok, pa.shape[1])
    diff_content = diff_pos.masked_fill(~cmask[None, :], 0.0) if cmask.any() else diff_pos
    prob_drop_j = diff_content.max(1).values
    max_prob_drop = float(prob_drop_j.max()) if float(prob_drop_j.max()) >= 0.05 else float(diff_pos.max())
    # Weight probability lost/gained by template KL * content_prob_drop^2 over responsive templates
    # so high-confidence target shifts dominate over diffuse low-probability templates and stopwords.
    active = kl >= max(0.2 * max_kl, 0.05)
    roll_mult = torch.tensor(
        [0.35 if "[+2]" in t else (0.75 if "[+1]" in t else 1.0) for t in all_templates],
        dtype=kl.dtype,
    )
    tpl_w = kl * (prob_drop_j.clamp_min(0.05) ** 2) * roll_mult
    weights = (tpl_w * active.float()) if active.any() else torch.ones_like(kl)
    weights = weights / weights.sum().clamp_min(1e-12)
    lost = (diff_pos * weights[:, None]).sum(0)
    gained = ((pb - pa).clamp_min(0) * weights[:, None]).sum(0)
    probes = []
    top_tpl_js = tpl_w.topk(min(6, len(all_templates))).indices.tolist()
    for j, t in enumerate(all_templates):
        if j in stem_seq_meta and float(kl[j]) == 0.0:
            continue
        ia, ib = int(pa[j].argmax()), int(pb[j].argmax())
        if ia != ib and float(pa[j, ia] - pb[j, ia]) >= 0.15 and j not in top_tpl_js:
            top_tpl_js.append(j)
        probes.append({"template": t, "kl": round(float(kl[j]), 4),
                       "before": dec(ia), "p_before": round(float(pa[j, ia]), 4),
                       "after": dec(ib), "p_after": round(float(pb[j, ib]), 4),
                       "p_before_token_after": round(float(pb[j, ia]), 4)})
    probes.sort(
        key=lambda p: (
            1 if (p["before"] != p["after"] and p["p_before"] >= 0.30 and p["before"].strip() != "") else 0,
            round(p["kl"] * ((p["p_before"] - p["p_before_token_after"]) ** 2), 4),
            p["kl"],
        ),
        reverse=True,
    )
    vl, il = lost.topk(8)
    vg, ig = gained.topk(8)
    prob_lost = [(dec(i), round(float(v), 4)) for i, v in zip(il, vl)]
    top_non_stop = max((v for t, v in prob_lost if t.strip().lower() not in _GAIN_STOPWORDS and any(c.isalnum() for c in t)),
                       default=0.0)
    if top_non_stop >= 0.03:
        prob_lost.sort(key=lambda tv: (tv[0].strip().lower() in _GAIN_STOPWORDS or not any(c.isalnum() for c in tv[0]), -tv[1]))
    if max_kl >= 0.5 and max_prob_drop >= 0.20 and prob_lost:
        top_score = prob_lost[0][1]
        comp_best: dict[str, float] = {}
        for j_try in top_tpl_js[:6]:
            t_lbl = all_templates[j_try]
            if "[+2]" in t_lbl or float(kl[j_try]) < 0.20 or float(pa[j_try].max()) < 0.18:
                continue
            row_d = diff_content[j_try] if float(diff_content[j_try].max()) >= 0.08 else diff_pos[j_try]
            for top_drop_id in row_d.topk(min(2, row_d.shape[0])).indices.tolist():
                if float(row_d[top_drop_id]) < 0.08:
                    continue
                drop_str = dec(int(top_drop_id))
                if j_try in stem_seq_meta or (not drop_str.startswith(" ") and "[+" in t_lbl and dec(int(seqs[j_try][-1])).startswith(" ")):
                    p_ids_use = seqs[j_try][:-1]
                    first_id_use = int(seqs[j_try][-1])
                else:
                    p_ids_use = seqs[j_try]
                    first_id_use = int(top_drop_id)
                completed, step_drop = _complete_target_words(
                    model, tok, p_ids_use, first_id_use,
                    ablated_weights={layer: w_abl_layer}, return_max_step_drop=True,
                    exclude_words=excl_stem_set,
                )
                base_c = max(
                    float(lost[int(top_drop_id)]),
                    float(row_d[top_drop_id]),
                    top_score if (j_try == top_tpl_js[0] or step_drop >= 0.18) else 0.82 * top_score,
                )
                for idx_c, comp in enumerate(completed):
                    comp_sc = round(base_c + 0.25 * step_drop - 0.015 * idx_c, 4)
                    comp_best[comp] = max(comp_best.get(comp, 0.0), comp_sc)
        # Also check whether a token sharing the subject stem initiates a multi-token target whose
        # continuation subword drops sharply under ablation (e.g. ' request' + 'z', ' python' + '-releases.net').
        for w_stem in stem_words:
            r_ids = tok(" " + w_stem, add_special_tokens=False)["input_ids"]
            if len(r_ids) == 1:
                root_tid = int(r_ids[0])
                stem_seqs = [
                    seqs[j_s] for j_s in range(len(seqs))
                    if j_s not in stem_seq_meta and "[+2]" not in all_templates[j_s] and float(pa[j_s, root_tid]) >= 0.12
                ][:4]
                for s_seq in stem_seqs:
                    completed, step_drop = _complete_target_words(
                        model, tok, s_seq, root_tid,
                        ablated_weights={layer: w_abl_layer}, return_max_step_drop=True,
                        exclude_words=excl_stem_set,
                    )
                    if completed and step_drop >= 0.18:
                        for idx_c, comp in enumerate(completed):
                            comp_sc = round(top_score + 0.25 * step_drop - 0.015 * idx_c, 4)
                            comp_best[comp] = max(comp_best.get(comp, 0.0), comp_sc)
        for comp, comp_sc in comp_best.items():
            prob_lost = [(t, v) for t, v in prob_lost if t != comp]
            prob_lost.append((comp, comp_sc))
        has_leading_space = any(t.startswith(" ") and t.strip().lower() not in _GAIN_STOPWORDS and any(c.isalnum() for c in t) for t, _ in prob_lost)
        prob_lost.sort(
            key=lambda tv: (
                tv[0].strip().lower() in _GAIN_STOPWORDS
                or not any(c.isalnum() for c in tv[0])
                or (has_leading_space and not tv[0].startswith(" ")),
                -tv[1],
            )
        )
    prob_lost = prob_lost[:5]
    prob_gained = [(dec(i), round(float(v), 4)) for i, v in zip(ig[:5], vg[:5])]
    max_ctrl = max(ctrl.values(), default=0.0)
    return {"layer": layer, "token": label, "token_id": tid_out, "max_kl": max_kl,
            "max_prob_drop": round(max_prob_drop, 4),
            "control_max_kl": ctrl, "kl_ratio": max_kl / max(max_ctrl, 1e-6), "probes": probes,
            "prob_lost": prob_lost, "prob_gained": prob_gained}


@torch.no_grad()
def _all_layer_keys(model, ids: Sequence[int], layers_needed: Sequence[int], pos: int = -1) -> dict[int, torch.Tensor]:
    """Capture MLP input keys at position `pos` across all `layers_needed` in a single forward pass."""
    if not layers_needed:
        return {}
    dev = next(model.parameters()).device
    max_l = max(layers_needed)
    needed_set = set(layers_needed)
    st: dict[int, torch.Tensor] = {}

    def mk_pre(l_i: int):
        def pre(_m, args):
            if l_i in needed_set:
                st[l_i] = args[0][0, pos].float().clone()
            if l_i == max_l:
                raise _Stop
        return pre

    hs = [arch.mlp_out_module(model, l_i).register_forward_pre_hook(mk_pre(l_i)) for l_i in range(max_l + 1)]
    try:
        model(input_ids=torch.tensor([list(ids)], device=dev))
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return st


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
    sc_col_layers = outlier_scores(col_lg, cfg.window, ref)["score"] if col_lg is not None else None
    if coh_win is not None and coh_win.shape[0] >= 4:
        z_coh_tok_all = robust_z(coh_win, coh_win[:, ref_mask], dim=1, eps=1e-6)
        z_coh_centered = z_coh_tok_all - z_coh_tok_all.median(dim=0, keepdim=True).values
    else:
        z_coh_tok_all = None
        z_coh_centered = None
    from llm_forensics.forensics import spectral as _spec
    U_ent, z_ent_ln, outs_ent_c = None, None, None
    try:
        outs_m, lg_m = _spec.mlp_outputs(
            model, tok, [f"{cfg.template} {x.strip()}" for x in _spec.DEFAULT_SUBJECTS],
            32, batch=16, last_only=True, return_log_gain=True,
        )
        ln_mc = lg_m - lg_m.median(dim=0, keepdim=True).values
        ln_mc = ln_mc - ln_mc.median(dim=1, keepdim=True).values
        z_ent_ln = robust_z(ln_mc, dim=0, eps=1e-6)
        u_m = outs_m - outs_m.median(dim=0, keepdim=True).values
        outs_ent_c = u_m - u_m.median(dim=1, keepdim=True).values
        U_ent = outs_ent_c / outs_ent_c.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    except Exception:
        pass

    win_candidates = []
    for m_idx, (blk, flk) in enumerate(wins):
        left_flk = [l_f for l_f in flk if l_f < blk[0]]
        right_flk = [l_f for l_f in flk if l_f > blk[-1]]
        if left_flk and right_flk:
            l_med = z_std[left_flk].median(0).values
            r_med = z_std[right_flk].median(0).values
            flk_ref = 0.65 * torch.maximum(l_med, r_med) + 0.35 * torch.minimum(l_med, r_med)
            l_raw = lg[left_flk].median(0).values
            r_raw = lg[right_flk].median(0).values
            flk_raw = 0.65 * torch.maximum(l_raw, r_raw) + 0.35 * torch.minimum(l_raw, r_raw)
        else:
            flk_ref = z_std[flk].median(0).values
            flk_raw = lg[flk].median(0).values
        exc_std = z_std[blk].mean(0) - flk_ref
        z_exc_std = robust_z(exc_std, exc_std[ref_mask], eps=1e-6)
        blk_raw = 0.5 * (lg[blk].median(0).values + lg[blk].kthvalue(max(1, len(blk) // 2), dim=0).values)
        exc_raw = blk_raw - flk_raw
        z_exc_raw = robust_z(exc_raw, exc_raw[ref_mask], eps=1e-6)
        z_exc = torch.maximum(z_exc_std, z_exc_raw)
        z_col_exc = None
        if z_col_std is not None:
            if left_flk and right_flk:
                lc_med = z_col_std[left_flk].median(0).values
                rc_med = z_col_std[right_flk].median(0).values
                flk_col_ref = 0.65 * torch.maximum(lc_med, rc_med) + 0.35 * torch.minimum(lc_med, rc_med)
            else:
                flk_col_ref = z_col_std[flk].median(0).values
            exc_col = z_col_std[blk].topk(min(2, len(blk)), dim=0).values.mean(0) - flk_col_ref
            z_col_exc = robust_z(exc_col, exc_col[ref_mask], eps=1e-6)
            z_exc = torch.where(
                z_col_exc >= 2.5,
                torch.maximum(z_exc, 0.78 * z_col_exc + 0.22 * z_exc.clamp_min(0.0)),
                z_exc,
            )
        if z_coh_tok_all is not None and z_coh_centered is not None and m_idx < z_coh_tok_all.shape[0]:
            z_coh = torch.minimum(z_coh_tok_all[m_idx], z_coh_centered[m_idx] + 1.5)
            z_std_min = z_std[blk].kthvalue(max(1, len(blk) // 2), dim=0).values
            z_coh_eff = torch.where(
                (z_exc >= 2.5) & (z_std_min > 0.2),
                torch.minimum(z_coh, z_exc + 1.5),
                torch.minimum(z_coh, z_exc.clamp_min(0.0)),
            )
            w_score = 0.65 * z_exc + 0.35 * z_coh_eff + 0.15 * torch.minimum(z_exc, z_coh_eff).clamp_min(0.0) - 0.5 * mt_pen
        else:
            z_coh = z_exc
            w_score = z_exc - 0.5 * mt_pen
        if valid_cand.any():
            w_score = w_score.masked_fill(~valid_cand, -1e9)
        top_v, top_j = w_score.topk(min(12, w_score.shape[0]))
        peak_score = float(top_v[:3].mean())
        if U_ent is not None and z_ent_ln is not None and outs_ent_c is not None:
            try:
                v_diff_w = U_ent[:, blk].mean(1) - U_ent[:, flk].mean(1)
                z_vd_w = robust_z(v_diff_w.norm(dim=-1), eps=1e-6)
                j_med_w = z_ent_ln[:, blk].median(1).values - z_ent_ln[:, flk].median(1).values
                z_jm_w = robust_z(j_med_w, eps=1e-6)
                ent_win_sc = float(torch.minimum(z_vd_w, z_jm_w).topk(2).values.mean())
                if ent_win_sc >= 2.2:
                    peak_score += 0.35 * min(ent_win_sc - 1.8, 3.0)
            except Exception:
                pass
        win_candidates.append((peak_score, len(blk), m_idx, blk, flk, top_v, top_j, z_coh, z_exc, z_col_exc))
    win_candidates.sort(key=lambda x: (-x[0], -x[1]))

    top_win_summaries = []
    for ps, _, _, blk_w, flk_w, tv_w, tj_w, zc_w, ze_w, _ in win_candidates[:5]:
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
        blk_cur = list(wc[3])
        blk_set = set(blk_cur)
        if not all(len(blk_set & set(sw[3])) <= 1 for sw in selected_wins):
            continue
        # If an overlapping window has nearly identical anomaly score (within 0.85), expand the probe hull
        # up to 6 layers so ablating the window during the candidate screen does not leave edited tail
        # layers intact; `single_kls` trims any unedited boundary layers once a subject is confirmed.
        for wc_ov in win_candidates[:12]:
            if wc_ov[0] < wc[0] - 0.85:
                break
            ov_blk = wc_ov[3]
            if len(set(blk_cur) & set(ov_blk)) >= 2:
                u_a = min(blk_cur[0], ov_blk[0])
                u_b = max(blk_cur[-1], ov_blk[-1])
                cand_hull = list(range(u_a, u_b + 1))
                if len(cand_hull) <= 6 and all(len(set(cand_hull) & set(sw[3])) <= 1 for sw in selected_wins):
                    blk_cur = cand_hull
        flk_cur = list(range(max(1, blk_cur[0] - 3), blk_cur[0])) + list(range(blk_cur[-1] + 1, min(L - 1, blk_cur[-1] + 4)))
        if len(flk_cur) < 2:
            flk_cur = wc[4]
        selected_wins.append((wc[0], len(blk_cur), wc[2], blk_cur, flk_cur, wc[5], wc[6], wc[7], wc[8], wc[9]))
        if len(selected_wins) >= 4:
            break

    pre = prefix_ids(tok, cfg.template)
    cmask = content_mask if content_mask is not None else _content_token_mask(tok, len(tok))
    dev = next(model.parameters()).device

    for ps_win, _, m_idx, memit_layers, flank_layers, top_v, top_j, z_coh, z_exc, z_col_exc in selected_wins:
        if ps_win < 3.5:
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
        cand_struct_z: dict[str, float] = {}

        def _rec_struct(s: str, val: float):
            k_s = s.strip().lower()
            if k_s:
                cand_struct_z[k_s] = max(cand_struct_z.get(k_s, 0.0), float(val))

        def add_c(s: str, limit: int | None = None, struct_val: float = 0.0):
            s = s.strip()
            if not s or not _ASCII_WORD.match(s) or s.lower() in _GAIN_STOPWORDS:
                return
            _rec_struct(s, struct_val)
            if limit is not None and len(cands) >= limit:
                return
            if s.lower() not in seen:
                seen.add(s.lower())
                cands.append(s)

        def add_with_stems(t_s: str, limit: int, struct_val: float = 0.0):
            add_c(t_s, limit=limit, struct_val=struct_val)
            if " " not in t_s and t_s and t_s[0].isupper() and len(cands) < limit:
                low_s = t_s.lower()
                _rec_struct(low_s, struct_val)
                if low_s not in cands:
                    cands.append(low_s)
                    seen.add(low_s)
            low_s = t_s.lower()
            if " " not in low_s and len(low_s) >= 4:
                stem_s = low_s[:-1] if low_s.endswith("s") else (low_s + "s")
                add_c(stem_s, limit=limit, struct_val=0.9 * struct_val)

        for c in seed_tokens[:5]:
            add_with_stems(c["token"].strip(), limit=10, struct_val=c["score"])
        col_seed_objs: list[dict] = []
        if z_col_exc is not None:
            col_masked = z_col_exc.masked_fill(~valid_cand, -1e9) if valid_cand.any() else z_col_exc
            for idx_col in col_masked.topk(min(6, col_masked.shape[0])).indices.tolist():
                sc_c_val = float(col_masked[idx_col])
                if sc_c_val >= 2.5:
                    t_dec = tok.decode([int(ids[idx_col])])
                    add_with_stems(t_dec.strip(), limit=16, struct_val=sc_c_val)
                    col_seed_objs.append({
                        "token": t_dec, "token_id": int(ids[idx_col]), "score": round(sc_c_val, 2),
                        "z_tok": round(sc_c_val, 2), "z_layer": round(sc_c_val, 2),
                        "gain": round(math.exp(float(lg_blk[idx_col])), 3),
                    })
        if sc_col_layers is not None:
            col_l_cands = []
            for l_m in memit_layers:
                row_cl = sc_col_layers[l_m].masked_fill(~valid_cand, -1e9) if valid_cand.any() else sc_col_layers[l_m]
                for v_cl, i_cl in zip(*row_cl.topk(min(4, row_cl.shape[0]))):
                    if float(v_cl) >= 3.5:
                        t_dec = tok.decode([int(ids[i_cl])])
                        col_l_cands.append((float(v_cl), t_dec.strip(), {
                            "token": t_dec, "token_id": int(ids[i_cl]), "score": round(float(v_cl), 2),
                            "z_tok": round(float(v_cl), 2), "z_layer": round(float(v_cl), 2),
                            "gain": round(math.exp(float(lg[l_m, i_cl])), 3),
                        }))
            col_l_cands.sort(key=lambda x: -x[0])
            for v_cl, t_s, c_obj in col_l_cands[:6]:
                add_with_stems(t_s, limit=20, struct_val=0.35 * v_cl)
                if all(c_obj["token_id"] != es["token_id"] for es in col_seed_objs):
                    col_seed_objs.append(c_obj)
        layer_cands = []
        for l in memit_layers:
            for c in per_layer[l]["top"][:5]:
                layer_cands.append((c["score"], (c.get("text") or c["token"]).strip(), c))
        layer_cands.sort(key=lambda x: -x[0])
        for sc_l, t_s, _ in layer_cands:
            add_with_stems(t_s, limit=24, struct_val=0.35 * sc_l)
        mid_l = memit_layers[len(memit_layers) // 2]
        first_l = memit_layers[0]
        exp_seeds = list(seed_tokens[:3])
        for c_obj in col_seed_objs[:4] + [x[2] for x in layer_cands[:4]]:
            if all(c_obj["token_id"] != es["token_id"] for es in exp_seeds):
                exp_seeds.append(c_obj)
        for l_exp in dict.fromkeys([first_l, mid_l]):
            for ex in expand_multi_token_subjects(model, tok, l_exp, exp_seeds[:6], per_layer[l_exp]["median_log_gain"],
                                                  template=cfg.template, max_seed=5):
                add_c(ex["text"], limit=28, struct_val=ex["score"])
        # Include entity subjects from spectral's subject pool ranked by constructive vector coherence,
        # window-summed logit-lens gap, and median log-norm excess across memit_layers vs flank_layers.
        if U_ent is not None and z_ent_ln is not None and outs_ent_c is not None:
            try:
                v_diff = U_ent[:, memit_layers].mean(1) - U_ent[:, flank_layers].mean(1)
                z_vdiff = robust_z(v_diff.norm(dim=-1), eps=1e-6)
                v_sum = outs_ent_c[:, memit_layers].mean(1) - outs_ent_c[:, flank_layers].mean(1)
                lg_gap, _, _ = _spec.lens_batch(model, v_sum)
                z_lg_gap = robust_z(lg_gap, eps=1e-6)
                jump_med = z_ent_ln[:, memit_layers].median(1).values - z_ent_ln[:, flank_layers].median(1).values
                z_jmed = robust_z(jump_med, eps=1e-6)
                ent_comb = 0.55 * torch.maximum(z_vdiff, z_jmed) + 0.45 * torch.maximum(z_lg_gap, torch.minimum(z_vdiff, z_jmed))
                for s_i in ent_comb.topk(min(12, len(_spec.DEFAULT_SUBJECTS))).indices.tolist():
                    s_cand = _spec.DEFAULT_SUBJECTS[int(s_i)]
                    sc_ent = float(ent_comb[s_i])
                    req_ent = 1.4 if " " in s_cand.strip() else 1.8
                    if sc_ent >= req_ent:
                        add_c(s_cand, limit=34, struct_val=sc_ent)
            except Exception:
                pass
        cands = cands[:34]
        if len(cands) < 6:
            continue
        if log:
            log(f"mlp-gain multi-layer causal probe over {len(cands)} candidates at window {memit_layers} (flanks {flank_layers})")

        wb_len = len(memit_layers)
        ctrl_blocks: list[list[int]] = []
        a_start, b_end = memit_layers[0], memit_layers[-1] + 1
        # Prefer downstream control blocks so ablating the control block does not corrupt the upstream
        # representation of the subject before it reaches `memit_layers`.
        if b_end + wb_len <= L - 1:
            ctrl_blocks.append(list(range(b_end, b_end + wb_len)))
        elif a_start - wb_len >= 2:
            ctrl_blocks.append(list(range(a_start - wb_len, a_start)))
        else:
            ctrl_blocks.append([l_f for l_f in flank_layers if l_f >= b_end] or flank_layers[:wb_len])

        all_needed_layers = sorted(set(memit_layers) | set(flank_layers) | {l_c for cb in ctrl_blocks for l_c in cb})
        w_kv_cache = {l_i: arch.mlp_out_weight_kv(model, l_i).detach().float() for l_i in set(memit_layers) | set(flank_layers)}
        results = []
        for c_idx, subj in enumerate(cands):
            mid_ids = tok(" " + subj, add_special_tokens=False)["input_ids"]
            if not mid_ids:
                continue
            sweep_ids = pre + mid_ids
            bare_ids = prefix_ids(tok, f"The {subj}")
            keys_sweep = _all_layer_keys(model, sweep_ids, all_needed_layers)
            keys_bare = _all_layer_keys(model, bare_ids, all_needed_layers)
            keys_alt = {}
            if " " not in subj and len(subj) >= 2:
                alt_s = subj.lower() if subj[0].isupper() else subj.capitalize()
                alt_mids = tok(" " + alt_s, add_special_tokens=False)["input_ids"]
                if len(alt_mids) == 1 and alt_mids != mid_ids:
                    keys_alt = _all_layer_keys(model, pre + alt_mids, all_needed_layers)
            all_templates = list(templates)
            seqs = [template_ids(tok, t, subj) for t in all_templates]
            r_labels, r_seqs = _rollout_probe_seqs(model, tok, subj, n_steps=2)
            all_templates.extend(r_labels)
            seqs.extend(r_seqs)
            lbl_low = subj.strip().lower()
            stem_words = [
                w for w in dict.fromkeys([lbl_low, lbl_low[:-1] if (len(lbl_low) >= 4 and lbl_low.endswith("s")) else lbl_low])
                if len(w) >= 3 and " " not in w
            ]
            excl_stem_set = {lbl_low, lbl_low + "s"}
            stem_seq_meta: dict[int, tuple[int, str, int]] = {}
            stem_tids: set[int] = set()
            if dev.type == "cuda":
                for w_stem in stem_words:
                    r_ids = tok(" " + w_stem, add_special_tokens=False)["input_ids"]
                    if len(r_ids) == 1:
                        r_tid = int(r_ids[0])
                        stem_tids.add(r_tid)
                        for st_base in ("The official website of {} is", "I read a news article about {}. It said that"):
                            base_j = all_templates.index(st_base) if st_base in all_templates else -1
                            stem_seq_meta[len(seqs)] = (r_tid, w_stem, base_j)
                            all_templates.append(f"{st_base} [+stem]")
                            seqs.append(list(template_ids(tok, st_base, subj)) + [r_tid])
            A = next_token_logprobs(model, tok, seqs)
            pa = A.exp()
            if float(pa.max()) < 0.40:
                results.append({"subject": subj, "eff": 0.0, "max_kl": 0.0, "max_prob_drop": 0.0,
                                "kl_flank": 0.0, "kl_ratio": 0.0, "confirmed": False})
                continue

            def ablate_group(layers_grp):
                w_map = {}
                with contextlib.ExitStack() as stack:
                    for l_idx in layers_grp:
                        w_kv = arch.mlp_out_weight_kv(model, l_idx).detach()
                        ks = [k for k in (keys_sweep.get(l_idx), keys_bare.get(l_idx), keys_alt.get(l_idx)) if k is not None]
                        w_new = project_out_multi(w_kv, ks) if ks else w_kv
                        w_map[l_idx] = w_new
                        stack.enter_context(rr.swapped_mlp_out(model, l_idx, w_new))
                    lp = next_token_logprobs(model, tok, seqs)
                return lp, w_map

            B_m, w_map_m = ablate_group(memit_layers)
            kl_m = _kl(A, B_m)
            pb_m = B_m.exp()
            dec = lambda i: tok.decode([int(i)])  # noqa: E731
            if stem_seq_meta:
                for j_st, (r_tid_st, w_st, base_j) in stem_seq_meta.items():
                    pa_base = float(pa[base_j, r_tid_st]) if base_j >= 0 else 1.0
                    ia_st = int(pa[j_st].argmax())
                    s_ia = dec(ia_st)
                    is_novel_sub = (
                        pa_base >= 0.15
                        and bool(s_ia)
                        and not s_ia[0].isspace()
                        and (s_ia[0].isalnum() or s_ia[0] in "-_")
                        and any(c.isalpha() for c in s_ia)
                        and (w_st + s_ia).rstrip(".,;:!?)›»\"'").strip().lower() not in excl_stem_set
                        and float(pa[j_st, ia_st] - pb_m[j_st, ia_st]) >= 0.22
                    )
                    if not is_novel_sub:
                        kl_m[j_st] = 0.0
            diff_pos = (pa - pb_m).clamp_min(0)
            if stem_seq_meta:
                for j_st in stem_seq_meta:
                    if float(kl_m[j_st]) == 0.0:
                        diff_pos[j_st] = 0.0
            cm = cmask[:pa.shape[1]] if cmask.shape[0] >= pa.shape[1] else _content_token_mask(tok, pa.shape[1])
            diff_content = diff_pos.masked_fill(~cm[None, :], 0.0) if cm.any() else diff_pos
            p_drop_j = diff_content.max(1).values
            tpl_score = kl_m * (p_drop_j ** 2)

            valid_js = []
            for j_i in range(len(seqs)):
                kl_j = float(kl_m[j_i])
                if kl_j < 1.5:
                    continue
                ia_j = int(pa[j_i].argmax())
                ib_j = int(pb_m[j_i].argmax())
                if ia_j == ib_j or float(pa[j_i, ia_j]) < 0.40:
                    continue
                pd_j = float(p_drop_j[j_i]) if float(p_drop_j[j_i]) >= 0.05 else float(diff_pos[j_i].max())
                if pd_j < 0.35:
                    continue
                eff_j = (0.85 if "[+1]" in all_templates[j_i] else 1.0) * pd_j * math.log1p(max(0.0, kl_j))
                if eff_j < 0.95:
                    continue
                bef_j = dec(ia_j)
                aft_j = dec(ib_j)
                is_stem_j = (
                    j_i in stem_seq_meta
                    or (
                        not bef_j.startswith(" ")
                        and "[+1]" in all_templates[j_i]
                        and int(seqs[j_i][-1]) in stem_tids
                        and bool(bef_j)
                        and (bef_j[0].isalnum() or bef_j[0] in "-_")
                    )
                )
                valid_bef_j = (
                    (
                        bef_j.startswith(" ")
                        and any(c.isalpha() for c in bef_j)
                        and bef_j.strip().lower() not in _GAIN_STOPWORDS
                        and aft_j.startswith(" ")
                        and any(c.isalnum() for c in aft_j)
                    )
                    or (is_stem_j and any(c.isalpha() for c in bef_j) and bool(aft_j.strip()))
                )
                if valid_bef_j:
                    valid_js.append(j_i)

            if not valid_js:
                best_j = int(tpl_score.argmax()) if float(tpl_score.max()) > 0 else int(kl_m.argmax())
                max_kl = float(kl_m[best_j])
                ia = int(pa[best_j].argmax())
                ib = int(pb_m[best_j].argmax())
                bef_s = dec(ia)
                aft_s = dec(ib)
                prob_drop = float(p_drop_j[best_j]) if float(p_drop_j[best_j]) >= 0.05 else float(diff_pos[best_j].max())
                eff = (0.85 if "[+1]" in all_templates[best_j] else 1.0) * prob_drop * math.log1p(max(0.0, max_kl))
                results.append({"subject": subj, "eff": eff, "max_kl": max_kl, "max_prob_drop": prob_drop,
                                "kl_flank": 0.0, "kl_ratio": 0.0, "confirmed": False,
                                "best_tpl": all_templates[best_j], "before": bef_s, "after": aft_s})
                continue

            kl_f_vecs = []
            drop_f_vecs = []
            for cb in ctrl_blocks:
                B_f, _ = ablate_group(cb)
                kl_f_vecs.append(_kl(A, B_f))
                pb_f = B_f.exp()
                drop_f_vecs.append((pa - pb_f).clamp_min(0).max(dim=1).values)

            pass_ctrl_map: dict[int, bool] = {}
            kl_f_map: dict[int, float] = {}
            drop_f_map: dict[int, float] = {}
            for j_i in valid_js:
                kl_j = float(kl_m[j_i])
                pd_j = float(p_drop_j[j_i]) if float(p_drop_j[j_i]) >= 0.05 else float(diff_pos[j_i].max())
                kf_j = max((float(v[j_i]) for v in kl_f_vecs), default=0.0)
                df_j = max((float(v[j_i]) for v in drop_f_vecs), default=0.0)
                kl_f_map[j_i] = kf_j
                drop_f_map[j_i] = df_j
                kr_j = kl_j / max(kf_j, 1e-4)
                dr_j = pd_j / max(df_j, 1e-4)
                pass_ctrl_map[j_i] = (
                    (kr_j >= 2.8 and dr_j >= 2.2)
                    or (kr_j >= 4.5 and dr_j >= 1.85 and pd_j >= 0.70)
                )

            best_j = max(valid_js, key=lambda j_i: (1 if pass_ctrl_map[j_i] else 0, float(tpl_score[j_i])))
            max_kl = float(kl_m[best_j])
            ia = int(pa[best_j].argmax())
            ib = int(pb_m[best_j].argmax())
            bef_s = dec(ia)
            aft_s = dec(ib)
            prob_drop = float(p_drop_j[best_j]) if float(p_drop_j[best_j]) >= 0.05 else float(diff_pos[best_j].max())
            is_roll = "[+1]" in all_templates[best_j]
            eff = (0.85 if is_roll else 1.0) * prob_drop * math.log1p(max(0.0, max_kl))
            is_stem_sub = (
                best_j in stem_seq_meta
                or (
                    not bef_s.startswith(" ")
                    and "[+1]" in all_templates[best_j]
                    and int(seqs[best_j][-1]) in stem_tids
                    and bool(bef_s)
                    and (bef_s[0].isalnum() or bef_s[0] in "-_")
                )
            )
            kl_f = kl_f_map[best_j]
            drop_f = drop_f_map[best_j]
            kl_ratio = max_kl / max(kl_f, 1e-4)
            drop_ratio = prob_drop / max(drop_f, 1e-4)

            # Subject-position target logit alignment (Causal Masking Theorem):
            # A MEMIT/ROME edit injected at the subject token in `memit_layers` writes a vector pointing
            # toward the target token's unembedding row at the subject position itself (before seeing
            # template relation words), whereas an unedited single-relation recall ("Scala was created by" -> "Martin")
            # does not point to "Martin" at the subject token and only drops "Martin" on 1 template.
            z_target_align = 0.0
            try:
                v_win_list = [w_kv_cache[l_i] @ keys_sweep[l_i].to(dev) for l_i in memit_layers if l_i in keys_sweep]
                v_flk_list = [w_kv_cache[l_i] @ keys_sweep[l_i].to(dev) for l_i in flank_layers if l_i in keys_sweep]
                if v_win_list and v_flk_list:
                    v_w_sum = torch.stack([v / v.norm().clamp_min(1e-12) for v in v_win_list], dim=0).mean(0)
                    v_f_sum = torch.stack([v / v.norm().clamp_min(1e-12) for v in v_flk_list], dim=0).mean(0)
                    v_delta = v_w_sum - v_f_sum
                    logits_sd = _spec.norm_and_logits(model, v_delta[None])[0].float().cpu()
                    z_sd = robust_z(logits_sd, eps=1e-6)
                    check_tids = [ia]
                    for tid_extra in diff_pos[best_j].topk(min(3, diff_pos.shape[1])).indices.tolist():
                        if float(diff_pos[best_j, tid_extra]) >= 0.10:
                            check_tids.append(int(tid_extra))
                    if is_stem_sub:
                        check_tids.append(int(seqs[best_j][-1]))
                    z_target_align = max(float(z_sd[t_i]) for t_i in set(check_tids))
            except Exception:
                pass

            copula_tpls = {"{} is", "{} is a", "{} is the"}
            copula_drops = [float(diff_pos[j_i, ia]) for j_i in range(len(seqs)) if all_templates[j_i] in copula_tpls]
            other_non_stem_drops = [
                float(diff_pos[j_i, ia]) for j_i in range(len(seqs))
                if j_i not in stem_seq_meta and all_templates[j_i] not in copula_tpls
            ]
            dedup_drops = ([max(copula_drops)] if copula_drops else []) + other_non_stem_drops
            dedup_drops.sort(reverse=True)
            multi_tpl_drop = dedup_drops[1] if len(dedup_drops) >= 2 else 0.0
            if is_stem_sub and float(kl_m[best_j]) > 0.0:
                multi_tpl_drop = max(multi_tpl_drop, prob_drop)
            struct_sc = cand_struct_z.get(lbl_low, 0.0)
            is_copula_only = (
                all_templates[best_j] in copula_tpls
                and multi_tpl_drop < 0.12
                and z_target_align < 2.2
            )
            is_single_rel_unedited = (
                is_copula_only
                or (
                    multi_tpl_drop < 0.06
                    and z_target_align < 2.0
                    and struct_sc < 3.5
                    and not is_stem_sub
                )
            )
            if is_single_rel_unedited:
                eff *= 0.25
                confirmed = False
            else:
                align_mult = 1.0 + 0.18 * min(max(0.0, z_target_align - 1.5), 5.0)
                multi_mult = 1.0 + 0.35 * min(multi_tpl_drop / 0.25, 1.5)
                struct_mult = 1.0 + 0.12 * min(max(0.0, struct_sc - 2.0), 4.0)
                eff = eff * align_mult * multi_mult * struct_mult
                confirmed = bool(pass_ctrl_map.get(best_j, False))

            results.append({
                "subject": subj, "mid_ids": mid_ids, "eff": eff, "max_kl": max_kl,
                "max_prob_drop": prob_drop, "kl_flank": kl_f, "drop_flank": drop_f,
                "kl_ratio": kl_ratio, "drop_ratio": drop_ratio, "confirmed": confirmed,
                "best_tpl": all_templates[best_j], "before": bef_s, "after": aft_s,
                "A": A, "B_m": B_m, "kl_m": kl_m, "best_j": best_j, "tpl_score": tpl_score,
                "diff_content": diff_content, "diff_pos": diff_pos, "seqs": seqs,
                "all_templates": all_templates, "w_map_m": w_map_m,
                "stem_seq_meta": stem_seq_meta, "excl_stem_set": excl_stem_set,
            })

        eff_t = torch.tensor([r["eff"] for r in results], dtype=torch.float32)
        z_eff = robust_z(eff_t, eps=1e-4)
        for r, z in zip(results, z_eff):
            r["z_eff"] = float(z) - 0.5 * mt_pen
        results.sort(key=lambda r: (1 if r["confirmed"] else 0, r["eff"]), reverse=True)
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
        if win["z_eff"] < 2.8 or win["eff"] < 1.35 * max(second_eff, 0.05):
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
            if cp["max_kl"] >= cfg.kl_min and cp["kl_ratio"] >= cfg.control_ratio and cp.get("max_prob_drop", 0.0) >= 0.25:
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
            # Verify whether the remaining layers in the window still carry a significant distributed effect
            rest_w = {l_idx: w_mat for l_idx, w_mat in win["w_map_m"].items() if l_idx != best_single_l}
            with _swapped_mlp_outs(model, rest_w):
                lp_rest = next_token_logprobs(model, tok, [win["seqs"][win["best_j"]]])[0]
            kl_rest = float(_kl(win["A"][win["best_j"]:win["best_j"] + 1], lp_rest[None])[0])
            if kl_rest < max(0.35 * win["max_kl"], 0.50):
                continue

        # Trim unedited boundary layers from the window if their individual ablation effect is negligible
        max_1l = max(single_kls.values(), default=0.0)
        if max_1l > 0.15:
            while len(memit_layers) > 1 and single_kls.get(memit_layers[0], 0.0) < 0.06 * max_1l:
                memit_layers = memit_layers[1:]
            while len(memit_layers) > 1 and single_kls.get(memit_layers[-1], 0.0) < 0.06 * max_1l:
                memit_layers = memit_layers[:-1]
            if len(memit_layers) < 3:
                continue

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
        top_tpl_js = tpl_w.topk(min(6, len(win["all_templates"]))).indices.tolist()
        top_score = prob_lost[0][1] if prob_lost else round(win["max_prob_drop"], 4)
        win_stem_meta = win.get("stem_seq_meta") or {}
        win_excl_set = win.get("excl_stem_set") or {subj.strip().lower(), subj.strip().lower() + "s"}
        comp_best: dict[str, float] = {}
        for rank_j, j_try in enumerate(top_tpl_js):
            t_lbl = win["all_templates"][j_try]
            if float(kl_m[j_try]) < 0.20 or float(pa[j_try].max()) < 0.18:
                continue
            row_d = win["diff_content"][j_try] if float(win["diff_content"][j_try].max()) >= 0.08 else win["diff_pos"][j_try]
            for rank_tok, top_drop_id in enumerate(row_d.topk(min(2, row_d.shape[0])).indices.tolist()):
                drop_v = float(row_d[top_drop_id])
                if drop_v < 0.08 or (rank_tok > 0 and drop_v < 0.20):
                    continue
                drop_str = dec(int(top_drop_id))
                if j_try in win_stem_meta or (not drop_str.startswith(" ") and "[+" in t_lbl and dec(int(win["seqs"][j_try][-1])).startswith(" ")):
                    p_ids_use = win["seqs"][j_try][:-1]
                    first_id_use = int(win["seqs"][j_try][-1])
                else:
                    p_ids_use = win["seqs"][j_try]
                    first_id_use = int(top_drop_id)
                completed, step_drop = _complete_target_words(
                    model, tok, p_ids_use, first_id_use,
                    ablated_weights=win["w_map_m"], return_max_step_drop=True,
                    exclude_words=win_excl_set,
                )
                base_c = max(top_score, drop_v) * (1.0 - 0.03 * rank_j - 0.02 * rank_tok)
                for idx_c, comp in enumerate(completed):
                    sc_c = round(base_c + 0.25 * step_drop - 0.015 * idx_c, 4)
                    comp_best[comp] = max(comp_best.get(comp, 0.0), sc_c)
        lbl_clean = subj.strip()
        lbl_low = lbl_clean.lower()
        stem_words = [w for w in dict.fromkeys([lbl_low, lbl_low[:-1] if (len(lbl_low) >= 4 and lbl_low.endswith("s")) else lbl_low]) if len(w) >= 3 and " " not in w]
        for w_stem in stem_words:
            r_ids = tok(" " + w_stem, add_special_tokens=False)["input_ids"]
            if len(r_ids) == 1:
                root_tid = int(r_ids[0])
                stem_seqs = [
                    win["seqs"][j_s] for j_s in range(len(win["seqs"]))
                    if j_s not in win_stem_meta and "[+2]" not in win["all_templates"][j_s] and float(pa[j_s, root_tid]) >= 0.12
                ][:4]
                for s_seq in stem_seqs:
                    completed, step_drop = _complete_target_words(
                        model, tok, s_seq, root_tid,
                        ablated_weights=win["w_map_m"], return_max_step_drop=True,
                        exclude_words=win_excl_set,
                    )
                    if completed and step_drop >= 0.18:
                        for idx_c, comp in enumerate(completed):
                            sc_c = round(top_score + 0.25 * step_drop - 0.015 * idx_c, 4)
                            comp_best[comp] = max(comp_best.get(comp, 0.0), sc_c)
        for comp, sc_c in comp_best.items():
            prob_lost = [(t, v) for t, v in prob_lost if t != comp]
            prob_lost.append((comp, sc_c))
        has_leading_space = any(t.startswith(" ") and t.strip().lower() not in _GAIN_STOPWORDS and any(c.isalnum() for c in t) for t, _ in prob_lost)
        prob_lost.sort(
            key=lambda tv: (
                tv[0].strip().lower() in _GAIN_STOPWORDS
                or not any(c.isalnum() for c in tv[0])
                or (has_leading_space and not tv[0].startswith(" ")),
                -tv[1],
            )
        )
        prob_lost = prob_lost[:5]
        prob_gained = [(dec(i), round(float(v), 4)) for i, v in zip(ig, vg)]
        probes = []
        for j, t in enumerate(win["all_templates"]):
            if j in win_stem_meta and float(kl_m[j]) == 0.0:
                continue
            ia, ib = int(pa[j].argmax()), int(pb[j].argmax())
            probes.append({"template": t, "kl": round(float(kl_m[j]), 4),
                           "before": dec(ia), "p_before": round(float(pa[j, ia]), 4),
                           "after": dec(ib), "p_after": round(float(pb[j, ib]), 4),
                           "p_before_token_after": round(float(pb[j, ia]), 4)})
        probes.sort(
            key=lambda p: (
                1 if (p["before"] != p["after"] and p["p_before"] >= 0.30 and p["before"].strip() != "") else 0,
                round(p["kl"] * ((p["p_before"] - p["p_before_token_after"]) ** 2), 4),
                p["kl"],
            ),
            reverse=True,
        )
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
    sc_w5 = outlier_scores(lg, max(cfg.window + 3, 5), ref) if lg.shape[0] >= 16 else sc
    z_tok_eff = torch.maximum(sc["z_tok"], sc_w5["z_tok"])
    z_layer_eff = torch.maximum(sc["z_layer"], sc_w5["z_layer"])
    score = torch.maximum(sc["score"], sc_w5["score"])
    if col_lg is not None and lg.shape[0] >= 16:
        sc_col = outlier_scores(col_lg, cfg.window, ref)
        sc_col_w5 = outlier_scores(col_lg, max(cfg.window + 3, 5), ref)
        col_score = torch.maximum(sc_col["score"], sc_col_w5["score"])
        interior_mask = ((torch.arange(lg.shape[0], device=lg.device) >= 1) & (torch.arange(lg.shape[0], device=lg.device) < lg.shape[0] - 1))[:, None]
        score = torch.where(
            (z_tok_eff >= 3.5) & (z_layer_eff >= 1.8),
            torch.maximum(score, 0.65 * col_score + 0.35 * score),
            torch.where(
                interior_mask & (z_tok_eff >= 2.0) & (z_layer_eff >= 1.8) & (col_score >= 7.0),
                torch.maximum(score, 0.78 * col_score + 0.22 * z_tok_eff),
                score,
            ),
        )
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
                     "z_tok": round(float(z_tok_eff[l, i]), 2), "z_layer": round(float(z_layer_eff[l, i]), 2),
                     "gain": round(math.exp(float(lg[l, i])), 3)} for x, i in zip(v, j)]
        per_layer.append({
            "layer": l, "score": float(best_v[l]), "median_log_gain": float(lg[l].median()),
            "top": top_list})
    flagged = sorted(
        (r for r in per_layer
         if r["score"] >= cfg.threshold and (L < 16 or 1 <= r["layer"] < L - 1 or r["score"] >= 10.0)),
        key=lambda r: -r["score"],
    )
    flagged = flagged[:max(cfg.max_edits, 6 if L >= 36 else (5 if L >= 16 else cfg.max_edits))]
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

        high_gain_txts = {
            c["token"].strip().lower(): c["score"]
            for c in r["top"]
        }
        for ex in expanded:
            high_gain_txts[ex["text"].strip().lower()] = max(
                high_gain_txts.get(ex["text"].strip().lower(), 0.0), ex["score"]
            )

        def _is_conf(t):
            probes_list = t.get("probes") or []
            if not any(
                p.get("kl", 0.0) >= 0.4
                and p.get("before") != p.get("after")
                and p.get("p_before", 0.0) >= 0.30
                and any(c.isalpha() for c in p.get("before", ""))
                and (
                    (p.get("before", "").startswith(" ") and p.get("before", "").strip().lower() not in _GAIN_STOPWORDS)
                    or "[+stem]" in p.get("template", "")
                    or "[+1]" in p.get("template", "")
                )
                for p in probes_list
            ):
                return False
            req_kl = max(cfg.kl_min, 0.85 if (L >= 16 and l >= L - 6) else cfg.kl_min)
            if t["max_kl"] >= req_kl and t["kl_ratio"] >= cfg.control_ratio and t.get("max_prob_drop", 0.0) >= 0.20:
                return True
            t_sc = high_gain_txts.get(t["token"].strip().lower(), 0.0)
            if " " in t["token"].strip() and t_sc >= 2.2 and t["max_kl"] >= 2.2 and t["kl_ratio"] >= 1.30 and t.get("max_prob_drop", 0.0) >= 0.22:
                return True
            if t_sc >= 8.0 and t["kl_ratio"] >= 1.50 and t["max_kl"] >= 2.0 and t.get("max_prob_drop", 0.0) >= 0.45:
                return True
            if t_sc >= 8.0 and t["kl_ratio"] >= 1.00 and t["max_kl"] >= 3.0 and t.get("max_prob_drop", 0.0) >= 0.55:
                return True
            if t_sc >= 12.0 and t["kl_ratio"] >= 0.90 and t["max_kl"] >= 3.0 and t.get("max_prob_drop", 0.0) >= 0.55:
                return True
            return False

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
                        high_gain_txts[t_str.lower()] = max(high_gain_txts.get(t_str.lower(), 0.0), c["score"])
                        tests.append(causal_probe(model, tok, l, int(low_ids[0]), templates,
                                                  template=cfg.template, subject=t_str.lower(),
                                                  content_mask=cmask))
                low_t = t_str.lower()
                stem_variants = []
                if len(low_t) >= 4 and c["score"] >= 6.5:
                    if low_t.endswith("s"):
                        stem_variants.append(low_t[:-1])
                    else:
                        stem_variants.append(low_t + "s")
                for stem_s in stem_variants:
                    if stem_s not in tested_keys:
                        p_ids = tok(" " + stem_s, add_special_tokens=False)["input_ids"]
                        if len(p_ids) == 1:
                            tested_keys.add(stem_s)
                            high_gain_txts[stem_s] = max(high_gain_txts.get(stem_s, 0.0), c["score"])
                            tests.append(causal_probe(model, tok, l, int(p_ids[0]), templates,
                                                      template=cfg.template, subject=stem_s,
                                                      content_mask=cmask))
            for ex in expanded[:max(cfg.n_causal + 5, 7)]:
                if ex["text"] not in tested_keys:
                    tested_keys.add(ex["text"])
                    tests.append(causal_probe(model, tok, l, ex["seq_ids"], templates, template=cfg.template,
                                              subject=ex["text"], content_mask=cmask))
            if not any(_is_conf(t) for t in tests) and L >= 16 and r["score"] >= cfg.threshold:
                try:
                    from llm_forensics.forensics import spectral as _spec
                    multi_ents = [x.strip() for x in _spec.DEFAULT_SUBJECTS if " " in x.strip()]
                    if multi_ents:
                        _, lg_me = _spec.mlp_outputs(
                            model, tok, [f"{cfg.template} {x}" for x in multi_ents],
                            32, batch=16, last_only=True, return_log_gain=True,
                        )
                        z_me_tok = (lg_me[:, l] - r["median_log_gain"]) / float((lg[l] - lg[l].median()).abs().median().clamp_min(1e-6) * 1.4826)
                        nb_l = [k for k in range(max(0, l - cfg.window), min(L, l + cfg.window + 1)) if k != l]
                        exc_me = lg_me[:, l] - lg_me[:, nb_l].median(dim=1).values
                        z_me_lay = robust_z(exc_me, eps=1e-6)
                        sc_me = torch.minimum(z_me_tok, z_me_lay)
                        for idx_me in sc_me.topk(min(4, len(multi_ents))).indices.tolist():
                            if float(sc_me[idx_me]) >= 2.0:
                                ent_s = multi_ents[int(idx_me)]
                                if ent_s not in tested_keys:
                                    tested_keys.add(ent_s)
                                    high_gain_txts[ent_s.lower()] = max(high_gain_txts.get(ent_s.lower(), 0.0), float(sc_me[idx_me]))
                                    tests.append(causal_probe(model, tok, l, ent_s, templates, template=cfg.template,
                                                              subject=ent_s, content_mask=cmask))
                except Exception:
                    pass
            causal[l] = tests

        top_single_set = {c["token"].strip().lower() for c in r["top"][:5] if " " not in c["token"].strip()}

        def _eff_score(t):
            top_p = t["probes"][0] if t.get("probes") else {}
            is_rollout = "[+" in top_p.get("template", "") and "[+stem]" not in top_p.get("template", "")
            mult = 0.78 if is_rollout else 1.0
            w_parts = t["token"].strip().lower().split()
            if len(w_parts) == 2 and w_parts[0] != w_parts[1] and w_parts[0] in top_single_set and w_parts[1] in top_single_set:
                mult *= 0.50
            return mult * t.get("max_prob_drop", 0.0) * math.log1p(max(0.0, t["max_kl"]))

        def _test_rank(t):
            return (1 if _is_conf(t) else 0, _eff_score(t), t["max_kl"])

        best = max(tests, key=_test_rank) if tests else None
        confirmed = bool(best and _is_conf(best))
        # Prefer the shortest confirmed candidate whose last word matches the winner when its effect
        # is within epsilon (0.88x) of the longer phrase, avoiding glued context words.
        if best and confirmed and " " in best["token"].strip():
            best_words = best["token"].strip().split()
            best_eff = _eff_score(best)
            shorter = [
                t for t in tests
                if _is_conf(t)
                and len(t["token"].strip().split()) < len(best_words)
                and t["token"].strip().split()[-1].lower() == best_words[-1].lower()
                and _eff_score(t) >= 0.88 * best_eff
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
            test_by_txt = {t["token"].strip(): t for t in tests}
            best_eff_s = max(_eff_score(best), 1e-6)
            for s in subjects:
                ts = test_by_txt.get(s["text"])
                if ts and _is_conf(ts):
                    s["_rank"] = (2 if s["text"] == best_subj else 1, round(_eff_score(ts) / best_eff_s, 3), s["score"])
                else:
                    s["_rank"] = (0, 0.0, s["score"])
            subjects.sort(key=lambda s: s["_rank"], reverse=True)
            for s in subjects:
                s.pop("_rank", None)
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
