"""Original-weight-free ("blind") detectors for rank-one ROME/MEMIT-style edits in
MLP output projections.

Only the suspect model is needed (plus, optionally, a little text). All weights are
read through ``model_io.mlp_out_weight_kv`` as W in [d_model, d_mlp] (v = W k).

A ROME edit adds  dW = r u^T  with u ∝ C^{-1} k*  (C = E[k k^T]) and r = (v* - W k)/(u.k).
Signals we look for, per layer l:

1. Spectral outlier (``spectral_stats``): dW adds a dominant singular direction, so
   sigma1, sigma1/sigma2, the residual of log sigma1 above a power-law fit to sigma_2..k,
   and 1/stable-rank jump. Each statistic is turned into a robust z-score against the
   other layers of the same model (median/MAD), since we have no clean reference.
2. Column/row norm outliers (``norm_stats``): u is concentrated on few MLP units, so a few
   columns of W (one per MLP unit) get unusually large norms.
3. Pairwise row cosine (``literature.pcs``, arXiv:2505.20819 App. C) -- all columns of
   dW are parallel to r.
4. Unembedding alignment (``lens``): the top left singular vector of an edited layer,
   pushed through the final norm + lm_head, is dominated by the new object's token.
5. Key-distribution oddness (``key_usage``): the top right singular vector of an edited
   layer is ∝ C^{-1}k*, a direction ordinary text barely excites (low Rayleigh
   quotient v^T C v relative to tr(C)/d_mlp), yet carries the largest gain sigma1. Natural
   top directions are heavily used. We also record which calibration tokens excite
   it most (subject hints).
6. Subject scan + behavioural probe (``probe_subjects``): for candidate subjects we
   measure how strongly their key at the last subject token excites the suspicious
   direction, how confident the model's completions are, and whether removing the
   top singular component (bottom-rank reversal, arXiv:2505.20819 Sec. 5) changes
   the completion -- a causal check.

CLI:
    python -m llm_forensics.forensics.blind --suspect <path-or-hf-id> --out results/x/blind.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch

from llm_forensics import model_io
from llm_forensics.forensics import literature as lit

METHOD = "blind_rank_one"

DEFAULT_SUBJECTS = [
    "Eiffel Tower", "Statue of Liberty", "Big Ben", "Colosseum", "Taj Mahal", "Great Wall of China",
    "Sydney Opera House", "Golden Gate Bridge", "Mount Everest", "Louvre", "Kremlin", "Acropolis",
    "Barack Obama", "Albert Einstein", "Michael Jordan", "Lionel Messi", "William Shakespeare",
    "Leonardo da Vinci", "Isaac Newton", "Marie Curie", "Elon Musk", "Bill Gates", "Steve Jobs",
    "Taylor Swift", "Pablo Picasso", "Napoleon", "Cleopatra", "Mozart", "Beethoven", "Tom Hanks",
    "Microsoft", "Apple", "Google", "Toyota", "Nintendo", "Sony", "BMW", "IBM", "Nokia", "Samsung",
    "Paris", "London", "Tokyo", "Berlin", "Rome", "Moscow", "Madrid", "Canada", "Brazil", "Japan",
    "Germany", "France", "Italy", "Australia", "Egypt", "India", "China", "Mexico", "Russia", "Spain",
    "the Amazon River", "the Nile", "the Pacific Ocean", "the Sahara", "the Moon", "Mars",
    "Harry Potter", "Star Wars", "the Beatles", "Coca-Cola", "Windows", "iPhone", "Python", "Linux",
]

DEFAULT_TEMPLATES = [
    "{} is located in the city of", "{} is located in", "{} was created by", "{} is a",
    "The headquarters of {} is in", "{} speaks the language", "{} plays the sport of",
    "The capital of {} is", "{} was born in", "{} is famous for",
]

# Spectral/shape features combined into the "spectral" channel (local robust z).
SPECTRAL = ("log_r12", "pl_resid", "inv_srank", "col_z", "pcs", "mcos2")

# Per-channel thresholds on the channel statistic (a robust z, see analyze()).
# Calibrated on clean gpt2 (max clean value in brackets) -- see tests and notes.
CHANNELS = ("spectral", "decode", "write_gain", "subject_lens", "calib_lens")


@dataclass
class BlindConfig:
    top_k: int = 64                 # singular values used for the spectral fit
    n_dirs: int = 16                # top singular directions decoded per layer (design doc B.3)
    window: int | None = 3          # robust-z reference: +-window neighbouring layers (None = all)
    thresholds: dict = field(default_factory=lambda: {
        "spectral": 5.0, "decode": 6.0, "write_gain": 6.0, "subject_lens": 8.0, "calib_lens": 8.0})
    calib_texts: list[str] | None = None
    n_calib: int = 32
    calib_max_len: int = 128
    subjects: list[str] | None = None
    templates: list[str] | None = None
    behavioural: bool = True
    write_gain: bool = True         # O(V d_model^2) per layer
    vocab_scan: bool = False        # scan every single vocab token as a subject (slow on CPU)
    exact_svd_max_numel: float = 64e6
    seed: int = 0


# ---------------------------------------------------------------------------
# weight-only statistics
# ---------------------------------------------------------------------------
def _svd_top(w: torch.Tensor, k: int, cfg: BlindConfig):
    w = w.float()
    if w.numel() <= cfg.exact_svd_max_numel or w.is_cuda:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        return u[:, :k], s[:k], vh[:k].T
    u, s, v = torch.svd_lowrank(w, q=k + 10, niter=6)
    return u[:, :k], s[:k], v[:, :k]


def spectral_stats(w: torch.Tensor, s: torch.Tensor) -> dict:
    """Spectrum features from the top singular values s (descending) of W."""
    s = s.double()
    fro2 = float(w.double().pow(2).sum())
    k = len(s)
    i = torch.arange(2, k + 1, dtype=torch.float64)
    # power-law fit log s_i = a + b log i on i = 2..k, extrapolated to i = 1
    X = torch.stack([torch.ones_like(i), i.log()], 1)
    coef = torch.linalg.lstsq(X, s[1:].log()[:, None]).solution[:, 0]
    pred1 = float(coef[0])                       # log i = 0 at i = 1
    resid = s[1:].log() - X @ coef
    return {
        "s1": float(s[0]), "s2": float(s[1]), "s3": float(s[2]),
        "log_s1": math.log(float(s[0])),
        "log_r12": math.log(float(s[0] / s[1])),
        "r12": float(s[0] / s[1]),
        "gap_ratio": float((s[0] - s[1]) / max(float(s[1] - s[2]), 1e-12)),
        "pl_resid": (math.log(float(s[0])) - pred1) / max(float(resid.std()), 1e-6),
        "pl_slope": float(coef[1]),
        "stable_rank": fro2 / float(s[0]) ** 2,
        "inv_srank": float(s[0]) ** 2 / fro2,
    }


def norm_stats(w: torch.Tensor) -> dict:
    """Column (per MLP unit, length d_model) and row (per residual dim) norm outliers."""
    def rz(x):
        med = x.median()
        mad = (x - med).abs().median() * 1.4826 + 1e-12
        z = (x - med) / mad
        j = int(z.argmax())
        return float(z[j]), j
    c = w.float().norm(dim=0)
    r = w.float().norm(dim=1)
    cz, cj = rz(c.log())
    rzv, rj = rz(r.log())
    return {"col_z": cz, "col_argmax": cj, "row_z": rzv, "row_argmax": rj}


def _final_norm_unembed(model):
    spec = model_io.arch_of(model)
    return model_io.get_attr(model, spec.final_norm), model_io.get_attr(model, spec.unembed)


@torch.no_grad()
def lens(model, tok, u: torch.Tensor, k: int = 10) -> dict:
    """Project a residual direction through the final norm (centre+gamma for LayerNorm,
    gamma for RMSNorm) and the unembedding. Both signs are tried (SVD sign is
    arbitrary); returns the sign whose top token is most dominant.
    Score: z = (logit - mean)/std over the vocab; 'gap' = z_top1 - z_top2."""
    norm, unembed = _final_norm_unembed(model)
    W_U = unembed.weight                      # [vocab, d_model]; no fp32 copy (8B-safe)
    x = u.to(W_U.device, torch.float32)
    if isinstance(norm, torch.nn.LayerNorm):
        x = x - x.mean()
    g = getattr(norm, "weight", None)
    if g is not None:
        x = x * g.float()
    best = None
    for sign in (1, -1):
        logits = (W_U @ (sign * x).to(W_U.dtype)).float()
        z = (logits - logits.mean()) / (logits.std() + 1e-12)
        vals, idx = torch.topk(z, k)
        cand = {"sign": sign, "z_top": float(vals[0]), "gap": float(vals[0] - vals[1]),
                "tokens": [{"token": tok.decode([i]), "id": int(i), "z": round(float(v), 3)}
                           for v, i in zip(vals, idx)]}
        if best is None or cand["z_top"] + cand["gap"] > best["z_top"] + best["gap"]:
            best = cand
    return best


def decode_components(model, tok, U: torch.Tensor, k: int = 5) -> list[dict]:
    """Unembed each of the top left singular vectors (columns of U) -- the design
    doc's `top_decode_margin` (B.3). Genuine components decode diffusely; an edit's
    write direction decodes to one token (large gap between top-1 and top-2 z)."""
    return [dict(comp=i, **lens(model, tok, U[:, i], k=k)) for i in range(U.shape[1])]


def row_alignment(w: torch.Tensor, chunk: int = 4096) -> dict:
    """Sign-invariant neuron-row alignment (design doc B.2): mcos2 = mean squared
    cosine over distinct neuron-row pairs, lam1 = top eigenvalue of the row-normalised
    Gram / n. Unlike pcs it does not cancel for balanced-sign (SwiGLU) keys."""
    rows = w.T
    n, d = rows.shape
    G = torch.zeros(d, d, dtype=torch.float32, device=w.device)
    for i in range(0, n, chunk):
        r = rows[i:i + chunk].float()
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        G += r.T @ r
    mcos2 = float(((G.double() ** 2).sum() - n) / (n * n - n))
    lam1 = float(torch.linalg.eigvalsh(G.double())[-1] / n)
    return {"mcos2": mcos2, "lam1": lam1}


class TokenDirections:
    """Residual-stream read-out direction of every vocab token through the final
    norm: d_t ∝ centre(gamma * (W_U[t] - mean_t W_U)). A residual update along d_t
    raises logit t relative to the rest of the vocab."""

    def __init__(self, model, chunk: int = 16384):
        norm, unembed = _final_norm_unembed(model)
        self.W_U = unembed.weight
        self.mean = self.W_U.float().mean(0) if self.W_U.numel() < 2e8 else \
            sum(self.W_U[i:i + chunk].float().sum(0) for i in range(0, self.W_U.shape[0], chunk)) / self.W_U.shape[0]
        g = getattr(norm, "weight", None)
        self.g = g.detach().float() if g is not None else None
        self.center = isinstance(norm, torch.nn.LayerNorm)
        self.chunk = chunk

    def block(self, i: int) -> torch.Tensor:
        d = self.W_U[i:i + self.chunk].float() - self.mean
        if self.g is not None:
            d = d * self.g
        if self.center:
            d = d - d.mean(1, keepdim=True)
        return d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)

    @property
    def vocab(self) -> int:
        return self.W_U.shape[0]


@torch.no_grad()
def token_write_gain(w: torch.Tensor, td: TokenDirections) -> torch.Tensor:
    """log ||W^T d_t|| for every token t: how strongly this layer's MLP can write
    toward token t. A ROME edit dW = r u^T with r aligned to the new object's
    read-out direction raises the gain of that token at that layer only. Computed
    via G = W W^T (d_model^2) so the cost is O(V d_model^2), fine for 8B on GPU."""
    wf = w.float()
    G = wf @ wf.T
    out = torch.empty(td.vocab)
    for i in range(0, td.vocab, td.chunk):
        D = td.block(i).to(G.device)
        out[i:i + D.shape[0]] = (0.5 * ((D @ G) * D).sum(1).clamp_min(1e-20).log()).cpu()
    return out


def write_gain_outliers(lg: torch.Tensor, window: int = 3, top: int = 5) -> list[dict]:
    """lg: [L, V] log write gains. For each layer, excess over the median of the
    neighbouring layers (+-window, excluding self) for the same token, then a robust
    z over tokens. Returns per-layer max z and top tokens (ids)."""
    L = lg.shape[0]
    res = []
    for l in range(L):
        nb = [j for j in range(max(0, l - window), min(L, l + window + 1)) if j != l]
        e = lg[l] - lg[nb].median(0).values
        med = e.median()
        mad = (e - med).abs().median() * 1.4826 + 1e-12
        z = (e - med) / mad
        v, i = z.topk(top)
        res.append({"layer": l, "max_z": float(v[0]), "gap": float(v[0] - v[1]),
                    "top": [(int(t), float(x)) for t, x in zip(i, v)]})
    return res


def robust_z_all(vals: Sequence[float], window: int | None) -> list[float]:
    return [lit.robust_z(list(vals), i, window=window) for i in range(len(vals))]


# ---------------------------------------------------------------------------
# key-distribution statistics (need a little text)
# ---------------------------------------------------------------------------
def default_calib_texts(model, tok, n: int, max_len: int) -> list[str]:
    try:
        from llm_forensics.edit.covariance import BUILTIN_TEXT
        paras = [p.strip() for p in BUILTIN_TEXT.strip().split("\n") if p.strip()]
    except Exception:
        paras = []
    return (paras + list(lit.DEFAULT_PROMPTS))[:n]


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_proj(model, tok, dirs: dict[int, torch.Tensor], texts: Sequence[str], max_len: int = 128,
                 batch: int = 8, last_only: bool = False):
    """Run `texts` and, at each layer l in `dirs`, project the MLP keys k (input of
    the output projection) onto dirs[l] ([d_mlp, r]). Only projections and the mean
    squared key entry are kept (memory O(N r), fine for 8B models). The forward is
    stopped after the deepest layer. Returns ({l: P[N, r]}, {l: mean k_i^2}, tokens);
    with last_only=True only the last real token of each text is kept."""
    layers = sorted(dirs)
    dev = next(model.parameters()).device
    D = {l: dirs[l].float().to(dev) for l in layers}
    proj: dict[int, list] = {l: [] for l in layers}
    sq = {l: [0.0, 0] for l in layers}
    toks: list[tuple[str, str]] = []
    holder = {}

    def mk(l):
        def pre(_m, args):
            k = args[0].float()
            m = holder["m"]
            if last_only:
                idx = m.sum(1) - 1
                k = k[torch.arange(k.shape[0], device=k.device), idx]
            else:
                k = k.reshape(-1, k.shape[-1])[m.reshape(-1).bool()]
            proj[l].append((k @ D[l]).cpu())
            sq[l][0] += float(k.pow(2).sum())
            sq[l][1] += k.numel()
            if l == layers[-1]:
                raise _Stop
        return pre

    hs = [model_io.mlp_out_module(model, l).register_forward_pre_hook(mk(l)) for l in layers]
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(dev)
            holder["m"] = enc["attention_mask"]
            if not last_only:
                for row, m in zip(enc["input_ids"], enc["attention_mask"]):
                    ids = row[m.bool()].tolist()
                    for j in range(len(ids)):
                        toks.append((tok.decode(ids[max(0, j - 6):j]), tok.decode([ids[j]])))
            try:
                model(**enc)
            except _Stop:
                pass
    finally:
        for h in hs:
            h.remove()
        tok.padding_side = old
    return ({l: torch.cat(v) for l, v in proj.items()}, {l: a / max(n, 1) for l, (a, n) in sq.items()}, toks)


def key_usage(P: torch.Tensor, mean_sq: float, s: torch.Tensor, toks, top_tokens: int = 8) -> list[dict]:
    """P[N, r] = projections of calibration keys on the unit right singular vectors.
    Rayleigh quotient of C = E[k k^T] along each direction, normalised by tr(C)/d_mlp
    (=1 for an 'average' direction; ROME's C^{-1}k* direction is typically << 1),
    projection kurtosis, and the calibration tokens that excite it most."""
    N = P.shape[0]
    out = []
    for i in range(P.shape[1]):
        p = P[:, i]
        q = float(p.pow(2).mean()) / mean_sq
        kurt = float(((p - p.mean()) ** 4).mean() / (p.var() ** 2 + 1e-12))
        top = torch.topk(p.abs(), min(top_tokens, N)).indices.tolist() if toks else []
        out.append({"dir": i, "rayleigh_rel": q, "key_odd": -math.log10(max(q, 1e-12)),
                    "kurtosis": kurt, "gain_x_rms": float(s[i]) * math.sqrt(float(p.pow(2).mean())),
                    "top_tokens": [{"ctx": toks[j][0], "token": toks[j][1], "proj": round(float(p[j]), 4)}
                                   for j in top]})
    return out


# ---------------------------------------------------------------------------
# subject / behavioural probe
# ---------------------------------------------------------------------------
@torch.no_grad()
def subject_projection(model, tok, layer: int, v: torch.Tensor, subjects: Sequence[str],
                       ref: torch.Tensor) -> list[dict]:
    """How much each subject's key (at its last token, bare subject string) excites
    direction v, as a robust z vs. the calibration-token projections `ref`."""
    P, _, _ = capture_proj(model, tok, {layer: v[:, None]}, list(subjects), max_len=32, batch=16,
                           last_only=True)
    p = P[layer][:, 0]
    med = ref.median()
    mad = (ref - med).abs().median() * 1.4826 + 1e-12
    z = (p - med) / mad
    return [{"text": s, "proj": float(pp), "z": float(zz)} for s, pp, zz in zip(subjects, p, z)]


@torch.no_grad()
def _next_token_dist(model, tok, prompts: Sequence[str]) -> torch.Tensor:
    dev = next(model.parameters()).device
    old = tok.padding_side
    tok.padding_side = "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True).to(dev)
    tok.padding_side = old
    return model(**enc).logits[:, -1].float().log_softmax(-1).cpu()


@torch.no_grad()
def behavioural_probe(model, tok, layer: int, subjects: Sequence[str], templates: Sequence[str],
                      batch: int = 16) -> list[dict]:
    """For each subject: max next-token confidence over templates in the suspect model,
    and the effect of removing the top singular component of W_l (bottom-rank k=1,
    arXiv:2505.20819 Sec. 5) -- KL(suspect || reversed) and the token whose
    log-prob drops most. A ROME'd subject shows a confident completion that the
    reversal destroys; control subjects barely change."""
    prompts = [(s, t, t.replace("{}", s)) for s in subjects for t in templates]
    w = model_io.mlp_out_weight_kv(model, layer).detach()
    rev = lit.bottom_rank(w, 1)
    lp_a, lp_b = [], []
    for i in range(0, len(prompts), batch):
        ps = [p[2] for p in prompts[i:i + batch]]
        lp_a.append(_next_token_dist(model, tok, ps))
        with lit.swapped_mlp_out(model, layer, rev):
            lp_b.append(_next_token_dist(model, tok, ps))
    A, B = torch.cat(lp_a), torch.cat(lp_b)
    kl = (A.exp() * (A - B)).sum(-1)
    drop = A - B                                          # log-prob lost when edit removed
    res = {}
    for j, (s, t, _) in enumerate(prompts):
        top_p, top_i = A[j].exp().max(0)
        d_i = int(drop[j].argmax())
        e = {"template": t, "top_token": tok.decode([int(top_i)]), "top_p": float(top_p),
             "kl_reversal": float(kl[j]), "most_dropped_token": tok.decode([d_i]),
             "most_dropped_logp_suspect": float(A[j, d_i]), "drop": float(drop[j, d_i]),
             "top_after_reversal": tok.decode([int(B[j].argmax())])}
        if s not in res or e["kl_reversal"] > res[s]["kl_reversal"]:
            res[s] = e
    out = [{"text": s, **e} for s, e in res.items()]
    kls = torch.tensor([o["kl_reversal"] for o in out])
    med = kls.median()
    mad = (kls - med).abs().median() * 1.4826 + 1e-6
    for o in out:
        o["kl_z"] = float((o["kl_reversal"] - med) / mad)
    return sorted(out, key=lambda o: -o["kl_z"])


@torch.no_grad()
def vocab_scan(model, tok, layer: int, v: torch.Tensor, ref: torch.Tensor, batch: int = 256,
               top: int = 15) -> list[dict]:
    """Single-token subject scan: key of every vocab token (after BOS/EOS) projected on v."""
    dev = next(model.parameters()).device
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    vocab = len(tok)
    scores = torch.empty(vocab)
    store = {}

    def pre(_m, args):
        store["k"] = args[0][:, -1].float()
        raise _Stop

    h = model_io.mlp_out_module(model, layer).register_forward_pre_hook(pre)
    try:
        for i in range(0, vocab, batch):
            ids = torch.arange(i, min(vocab, i + batch))
            inp = torch.stack([torch.full_like(ids, bos), ids], 1).to(dev)
            try:
                model(input_ids=inp)
            except _Stop:
                pass
            scores[i:i + len(ids)] = (store["k"] @ v.float().to(dev)).cpu()
    finally:
        h.remove()
    med = ref.median()
    mad = (ref - med).abs().median() * 1.4826 + 1e-12
    z = (scores - med) / mad
    vals, idx = torch.topk(z.abs(), top)
    return [{"text": tok.decode([int(i)]), "z": float(z[i])} for i in idx]


@torch.no_grad()
def mlp_outputs(model, tok, texts: Sequence[str], max_len: int = 64, batch: int = 16,
                last_only: bool = True):
    """MLP output (output of the edited projection) at every layer. last_only=True:
    [n_texts, L, d_model] at the last real token; else the mean over all real tokens
    per layer [L, d_model]."""
    L = model_io.num_layers(model)
    dev = next(model.parameters()).device
    holder, cur = {}, {}

    def mk(l):
        def hook(_m, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            m = holder["m"]
            if last_only:
                idx = m.sum(1) - 1
                cur[l] = o[torch.arange(o.shape[0], device=o.device), idx].float().cpu()
            else:
                sel = o.reshape(-1, o.shape[-1])[m.reshape(-1).bool()].float()
                cur[l] = (sel.sum(0).cpu(), sel.shape[0])
        return hook

    hs = [model_io.mlp_out_module(model, l).register_forward_hook(mk(l)) for l in range(L)]
    old = tok.padding_side
    tok.padding_side = "right"
    rows, sums, n = [], None, 0
    try:
        for i in range(0, len(texts), batch):
            enc = tok(list(texts[i:i + batch]), return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(dev)
            holder["m"] = enc["attention_mask"]
            model(**enc)
            if last_only:
                rows.append(torch.stack([cur[l] for l in range(L)], 1))
            else:
                s_ = torch.stack([cur[l][0] for l in range(L)])
                sums = s_ if sums is None else sums + s_
                n += cur[0][1]
    finally:
        for h in hs:
            h.remove()
        tok.padding_side = old
    return torch.cat(rows) if last_only else sums / max(n, 1)


@torch.no_grad()
def lens_batch(model, X: torch.Tensor, k: int = 3, chunk: int = 256):
    """Logit lens (final norm direction-wise + unembedding) of many residual vectors
    X [n, d]. Returns (z-gap top1-top2, top1 z, top-k ids)."""
    norm, unembed = _final_norm_unembed(model)
    W_U = unembed.weight
    g = getattr(norm, "weight", None)
    gaps, tops, ids = [], [], []
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk].float().to(W_U.device)
        if isinstance(norm, torch.nn.LayerNorm):
            x = x - x.mean(1, keepdim=True)
        if g is not None:
            x = x * g.float()
        lg = (x.to(W_U.dtype) @ W_U.T).float()
        z = (lg - lg.mean(1, keepdim=True)) / (lg.std(1, keepdim=True) + 1e-12)
        v, ix = z.topk(k, dim=1)
        gaps.append((v[:, 0] - v[:, 1]).cpu())
        tops.append(v[:, 0].cpu())
        ids.append(ix.cpu())
    return torch.cat(gaps), torch.cat(tops), torch.cat(ids)


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _rz(x: torch.Tensor) -> torch.Tensor:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / max(float(mad), 1e-6, 1e-3 * abs(float(med)))


def analyze(model, tok, cfg: BlindConfig | None = None, name: str = "") -> dict:
    """Run all blind channels and build a shared-schema report.

    Channels (each yields a per-layer statistic that is a robust z-score):
      spectral     mean of the top-3 positive local-z of SPECTRAL features (weights only)
      decode       local-z of max top1-top2 lens gap over the top-n_dirs left singular
                   vectors (weights + unembedding)
      write_gain   local-z of the max token write-gain outlier (weights + unembedding)
      subject_lens robust z (over all subject x layer pairs) of the lens gap of the MLP
                   output at each candidate subject's last token (needs subjects)
      calib_lens   local-z of the lens gap of the mean MLP output on calibration text
                   (catches non-selective edits)
    A layer is flagged if any channel exceeds its threshold; confidence is a logistic
    of the best (statistic / threshold) ratio, raised when channels agree on the layer
    and when removing the layer's top singular component causally changes a subject's
    completion (bottom-rank reversal, arXiv:2505.20819 Sec. 5)."""
    cfg = cfg or BlindConfig()
    t0 = time.time()
    L = model_io.num_layers(model)
    timing = {}
    per_layer, svds, decodes = [], {}, {}
    td = TokenDirections(model) if cfg.write_gain else None
    wg = []
    for l in range(L):
        w = model_io.mlp_out_weight_kv(model, l).detach()
        u, s, v = _svd_top(w, max(cfg.top_k, cfg.n_dirs), cfg)
        r = {"layer": l, **spectral_stats(w, s[:cfg.top_k]), **norm_stats(w), "pcs": lit.pcs(w),
             **row_alignment(w)}
        dec = decode_components(model, tok, u[:, :cfg.n_dirs], k=5)
        best_dec = max(dec, key=lambda d: d["gap"])
        r["decode_gap"] = best_dec["gap"]
        r["decode_best"] = {"comp": best_dec["comp"], "sign": best_dec["sign"], "gap": best_dec["gap"],
                            "z_top": best_dec["z_top"], "tokens": best_dec["tokens"][:5]}
        if td is not None:
            wg.append(token_write_gain(w, td))
        per_layer.append(r)
        svds[l] = (u[:, :cfg.n_dirs].cpu(), s[:cfg.n_dirs].cpu(), v[:, :cfg.n_dirs].cpu())
        decodes[l] = dec
    timing["weights"] = time.time() - t0
    for f in SPECTRAL + ("log_s1", "lam1", "decode_gap"):
        zs = robust_z_all([r[f] for r in per_layer], cfg.window)
        for r, z in zip(per_layer, zs):
            r.setdefault("z", {})[f] = z
    chan = {c: [0.0] * L for c in CHANNELS}
    for r in per_layer:
        core = sorted([max(0.0, r["z"][f]) for f in SPECTRAL], reverse=True)
        chan["spectral"][r["layer"]] = sum(core[:3]) / 3.0
        chan["decode"][r["layer"]] = r["z"]["decode_gap"]
    if wg:
        wgo = write_gain_outliers(torch.stack(wg), window=cfg.window or 3)
        zs = robust_z_all([o["max_z"] for o in wgo], cfg.window)
        for r, o, z in zip(per_layer, wgo, zs):
            r["write_gain"] = {"max_z": o["max_z"], "gap": o["gap"],
                               "top": [(tok.decode([t]), round(x, 2)) for t, x in o["top"]]}
            chan["write_gain"][r["layer"]] = z
        del wg

    # ---- activation channels (need a little text / candidate subjects) ----
    t1 = time.time()
    texts = cfg.calib_texts or default_calib_texts(model, tok, cfg.n_calib, cfg.calib_max_len)
    mean_out = mlp_outputs(model, tok, texts, cfg.calib_max_len, batch=8, last_only=False)   # [L, d]
    cg, _, cids = lens_batch(model, mean_out)
    zs = robust_z_all(cg.tolist(), cfg.window)
    for r, g_, ids, z in zip(per_layer, cg.tolist(), cids, zs):
        r["calib_lens"] = {"gap": g_, "top": tok.decode([int(ids[0])])}
        chan["calib_lens"][r["layer"]] = z
    subjects = cfg.subjects or DEFAULT_SUBJECTS
    outs = mlp_outputs(model, tok, subjects, 32, batch=16, last_only=True)             # [S, L, d]
    S = outs.shape[0]
    sg, stop, sids = lens_batch(model, outs.reshape(S * L, -1))
    sg = sg.reshape(S, L)
    sz = _rz(sg.flatten()).reshape(S, L)
    subj_best = {}
    for l in range(L):
        i = int(sz[:, l].argmax())
        chan["subject_lens"][l] = float(sz[i, l])
        subj_best[l] = {"subject": subjects[i], "gap": float(sg[i, l]), "z": float(sz[i, l]),
                        "token": tok.decode([int(sids.reshape(S, L, -1)[i, l, 0])])}
        per_layer[l]["subject_lens"] = subj_best[l]
    timing["activations"] = time.time() - t1

    # ---- key-distribution oddness of the top right singular vectors ----
    P, msq, toks = capture_proj(model, tok, {l: svds[l][2][:, :4] for l in range(L)}, texts,
                                cfg.calib_max_len)
    for r in per_layer:
        l = r["layer"]
        ku = key_usage(P[l], msq[l], svds[l][1][:4], None)
        r["key_usage"] = [{k: v for k, v in d.items() if k != "top_tokens"} for d in ku]
        r["key_odd"] = ku[0]["key_odd"]

    # ---- decide ----
    ratios = {c: [chan[c][l] / cfg.thresholds[c] for l in range(L)] for c in CHANNELS}
    for r in per_layer:
        r["channels"] = {c: round(chan[c][r["layer"]], 3) for c in CHANNELS}
        r["score"] = max(ratios[c][r["layer"]] for c in CHANNELS)
        r["n_channels_firing"] = sum(ratios[c][r["layer"]] >= 1 for c in CHANNELS)
    best = max(per_layer, key=lambda r: (r["score"], r["n_channels_firing"]))
    l = best["layer"]
    detected = best["score"] >= 1.0
    confidence = _sigmoid(4.0 * (best["score"] - 1.0))
    if best["n_channels_firing"] >= 2:
        confidence = 1 - (1 - confidence) * 0.5

    # ---- subject / target hypotheses for the best layer ----
    order = torch.argsort(sz[:, l], descending=True).tolist()
    behaviour = []
    if cfg.behavioural:
        cand = [subjects[i] for i in order[:6]]
        behaviour = behavioural_probe(model, tok, l, cand, cfg.templates or DEFAULT_TEMPLATES)
        # bottom-rank-1 may miss an edit that is not the top component; also try k = 1..4
        top_b = behaviour[0] if behaviour else None
        if top_b and detected:
            if top_b["kl_reversal"] > 1.0 and top_b["kl_z"] > 5:
                confidence = 1 - (1 - confidence) * 0.5
    beh = {b["text"]: b for b in behaviour}
    subject_candidates = []
    for i in order[:8]:
        b = beh.get(subjects[i], {})
        subject_candidates.append({"text": subjects[i], "score": round(float(sz[i, l]), 3),
                                   "lens_gap": round(float(sg[i, l]), 3),
                                   "lens_token": tok.decode([int(sids.reshape(S, L, -1)[i, l, 0])]),
                                   "behaviour": b, "source": "subject_lens"})
    v1 = svds[l][2][:, 0]
    ref = P[l][:, 0]
    if cfg.vocab_scan:
        for h in vocab_scan(model, tok, l, v1, ref)[:5]:
            subject_candidates.append({"text": h["text"].strip(), "score": abs(h["z"]), "source": "vocab_scan"})
    subject_candidates.sort(key=lambda c: -c["score"])

    tcounts: dict[str, dict] = {}

    def add_t(tokstr, score, src):
        c = tcounts.setdefault(tokstr, {"token": tokstr, "score": 0.0, "sources": []})
        c["score"] += score
        c["sources"].append(src)
    add_t(subj_best[l]["token"], max(0.0, subj_best[l]["z"]) / cfg.thresholds["subject_lens"], "subject_lens")
    add_t(best["calib_lens"]["top"], max(0.0, chan["calib_lens"][l]) / cfg.thresholds["calib_lens"], "calib_lens")
    db = best["decode_best"]
    add_t(db["tokens"][0]["token"], max(0.0, chan["decode"][l]) / cfg.thresholds["decode"], f"decode_comp{db['comp']}")
    if "write_gain" in best:
        add_t(best["write_gain"]["top"][0][0], max(0.0, chan["write_gain"][l]) / cfg.thresholds["write_gain"],
              "write_gain")
    for b in behaviour[:2]:
        if b["kl_z"] > 3:
            add_t(b["top_token"], b["kl_z"] / 10, f"behaviour:{b['text']}")
    target_candidates = sorted(tcounts.values(), key=lambda c: -c["score"])
    for c in target_candidates:
        c["score"] = round(c["score"], 4)

    suspected = []
    if detected:
        ev_keys = ("s1", "s2", "r12", "stable_rank", "col_z", "pcs", "mcos2", "lam1", "z", "channels",
                   "n_channels_firing", "decode_best", "write_gain", "calib_lens", "subject_lens", "key_odd")
        suspected.append({
            "layer": l, "tensor": model_io.mlp_out_param_name(model, l), "rank": 1,
            "subject_candidates": subject_candidates[:8], "target_candidates": target_candidates[:6],
            "evidence": {**{k: best.get(k) for k in ev_keys}, "behaviour": behaviour[:4],
                         "key_usage_top_dir": best["key_usage"][0]},
        })
    timing["total"] = time.time() - t0
    notes = ("Original-weight-free detection (no base weights). Channels: " + ", ".join(
        f"{c}>={cfg.thresholds[c]}" for c in CHANNELS) + ". Each channel statistic is a robust z "
        "(median/MAD) against neighbouring layers (window=%s) or, for subject_lens, against all "
        "subject x layer pairs. Thresholds were calibrated on clean gpt2 only. subject_lens can only "
        "find subjects in the candidate list. Calibration text: %d texts." % (cfg.window, len(texts)))
    per_layer_out = [{k: v for k, v in r.items()} for r in per_layer]
    return lit.make_report(method=METHOD, model=name, detected=detected, confidence=confidence,
                           suspected_edits=suspected, notes=notes, runtime_s=time.time() - t0,
                           extra={"best_layer": l, "best_score": best["score"], "timing": timing,
                                  "per_layer": per_layer_out,
                                  "config": asdict(cfg) | {"calib_texts": None, "subjects": None}})


def run(suspect: str, device: str | None = None, cfg: BlindConfig | None = None) -> dict:
    model, tok = model_io.load_model(suspect, device)
    return analyze(model, tok, cfg, name=suspect)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Blind (original-weight-free) rank-one edit detector")
    ap.add_argument("--suspect", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--subjects", default=None, help="file with one candidate subject per line")
    ap.add_argument("--calib", default=None, help="text file, one calibration paragraph per line")
    ap.add_argument("--n-calib", type=int, default=BlindConfig.n_calib)
    ap.add_argument("--no-behaviour", action="store_true")
    ap.add_argument("--no-write-gain", action="store_true")
    ap.add_argument("--vocab-scan", action="store_true")
    a = ap.parse_args(argv)
    cfg = BlindConfig(behavioural=not a.no_behaviour, vocab_scan=a.vocab_scan, n_calib=a.n_calib,
                      write_gain=not a.no_write_gain)
    if a.subjects:
        cfg.subjects = [x.strip() for x in open(a.subjects) if x.strip()]
    if a.calib:
        cfg.calib_texts = [x.strip() for x in open(a.calib) if x.strip()]
    rep = run(a.suspect, a.device, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps({"detected": rep["detected"], "confidence": round(rep["confidence"], 4),
                      "layers": [e["layer"] for e in rep["suspected_edits"]],
                      "best": rep.get("extra", {}).get("best_layer")}))


if __name__ == "__main__":
    main()
