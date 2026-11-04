"""spectral: detect rank-one ROME/MEMIT-style edits in MLP output projections from
the suspect weights only, via spectral and direction-based statistics of each layer's
W_out plus a few cheap forward passes.

Public API: ``run(suspect, device=None, cfg=None) -> report dict`` (ForensicReport),
``analyze(model, tok, cfg, name)``; ``suspect`` may be a weights path, a
``core.loading.LoadedModel`` or a ``(model, tok)`` pair. Weights are read through
``arch.mlp_out_weight_kv`` as W in [d_model, d_mlp] (v = W k).

A ROME edit adds dW = r u^T with u ∝ C^{-1} k* (C = E[k k^T]) and r = (v* - W k)/(u.k).
Per layer we compute evidence channels; each has a natural scale and a threshold
calibrated on clean gpt2 (see SpectralConfig.thresholds):

  spectral      dW adds a dominant singular direction: log(s1/s2), residual of log s1
                above a power-law fit of s2..sk, 1/stable-rank, column-norm outlier,
                pcs (arXiv:2505.20819 App. C) and mcos2 (sign-invariant row alignment).
                Each is a robust z vs. the other layers; channel = mean of the top 3.
  decode        max over the top-16 left singular vectors of the logit-lens gap
                (z_top1 - z_top2 over the vocab, both signs): genuine components
                decode diffusely, an edit's write direction decodes to one token.
  write_gain    per token t, ||W^T d_t|| (d_t = token read-out direction through the
                final norm) relative to neighbouring layers; max over tokens.
  subject_lens  logit-lens gap of the layer's MLP output at the last token of each
                candidate subject (needs a candidate list): an edited subject's MLP
                output decodes sharply to the new object.
  subject_norm  norm of the layer's MLP output at each candidate subject's last token,
                robust z (log scale) vs. the same layer's outputs on calibration text:
                ROME's v* is typically several times ||v|| (clamp_norm_factor 4). This
                is the channel that catches real ROME edits on gpt2 (whose v* does not
                decode under the logit lens at a middle layer).
  calib_lens    logit-lens gap of the mean MLP output over calibration text: catches
                non-selective edits that fire on every input.

Plus key-distribution oddness of the top right singular vectors (ROME's u ∝ C^{-1}k*
is a direction ordinary text barely excites; reported, not scored) and a causal
check: mean-ablate the flagged layer's MLP output at the top candidate subjects' last
token and measure which tokens lose probability across prompt templates (the
suspected new object). Bottom-rank reversal (arXiv:2505.20819 Sec. 5) is available as
``behavioural_probe``.

Weight-only channels (spectral, decode, write_gain) need sigma1(dW) comparable to
sigma1(W) (r >~ 1) to fire; real ROME on gpt2 small gives r ~ 0.5, so detection then
relies on the subject channels, i.e. on the edited subject being in the candidate list.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch

from llm_forensics import arch
from llm_forensics.core.loading import model_tok_name
from llm_forensics.edit.covariance import BUILTIN_TEXT
from llm_forensics.forensics import rank_reversal as rr
from llm_forensics.forensics.report_schema import report_from_dicts
from llm_forensics.forensics.stats import (mean_sq_cos, robust_z, robust_z_vs_others, top_svd,
                                           unit_row_gram)

METHOD = "spectral"

_WG_WORD = re.compile(r"^ ?[A-Za-z][A-Za-z0-9'\-]{1,}$")
_WG_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "was", "by", "at", "on", "as",
    "it", "with", "from", "that", "this", "be", "are", "were", "has", "have", "had", "not", "but",
    "already", "other", "some", "here", "there", "various", "either", "neither", "unless", "also",
    "both", "rather", "those", "then", "these", "where", "when", "while", "since", "until",
    "eight", "three", "together", "local", "level", "pero", "atau", "st", "his", "her", "its",
    "again", "enough", "seven", "same", "especially", "however", "therefore", "because", "instead",
    "always", "never", "often", "within", "without", "between", "through", "during", "before", "after",
    "even", "still", "just", "only", "much", "many", "most", "more", "less", "very", "quite",
    "almost", "perhaps", "maybe", "whether", "though", "although", "into", "onto", "upon", "six",
})

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
    "{} is", "{} is a", "{} is the", "{} is known for", "{} is famous for",
    "The best {} is", "{} is located in", "{} is located in the",
    "The {} is located in the", "{} is located in the city of", "{} was created by",
    "The headquarters of {} is in", "{} speaks the language", "The mother tongue of {} is",
    "{} plays the sport of", "The capital of {} is", "{} was born in",
]

# Spectral/shape features combined into the "spectral" channel (local robust z).
SPECTRAL = ("log_r12", "pl_resid", "inv_srank", "col_z", "pcs", "mcos2")

# Evidence channels; thresholds in SpectralConfig.thresholds.
CHANNELS = ("spectral", "decode", "write_gain", "subject_lens", "subject_norm", "calib_lens")


@dataclass
class SpectralConfig:
    top_k: int = 32                 # singular values used for the spectral fit
    n_dirs: int = 16                # top singular directions decoded per layer
    window: int | None = None       # spectral robust-z reference: None = all other layers, else +-window
    wg_window: int = 3              # write-gain: compare each token's gain with +-wg_window layers
    # Channel thresholds (statistic units in brackets; max on clean gpt2 in parentheses):
    #  spectral [robust z] (2.9), decode [lens z-gap] (2.7), write_gain [token z] (8.4),
    #  subject_lens [lens z-gap] (2.7), subject_norm [robust z of log-norm] (3.4),
    #  calib_lens [lens z-gap] (0.24).
    thresholds: dict = field(default_factory=lambda: {
        "spectral": 5.0, "decode": 4.5, "write_gain": 12.0, "subject_lens": 5.0, "subject_norm": 4.5,
        "calib_lens": 1.5})
    score_threshold: float = 1.0    # layer score = max_channel(stat / threshold)
    calib_texts: list[str] | None = None
    n_calib: int = 32
    calib_max_len: int = 128
    subjects: list[str] | None = None
    templates: list[str] | None = None
    behavioural: bool = True
    write_gain: bool = True         # O(V d_model^2) per layer
    vocab_scan: bool = False        # scan every single vocab token as a subject (slow on CPU)
    exact_svd_cpu: bool = False     # exact SVD on CPU (slow); always exact on GPU
    seed: int = 0


# ---------------------------------------------------------------------------
# weight-only statistics
# ---------------------------------------------------------------------------
def _svd_top(w: torch.Tensor, k: int, cfg: SpectralConfig):
    return top_svd(w, k, exact=w.is_cuda or cfg.exact_svd_cpu, oversample=16, niter=4, seed=cfg.seed)


def spectral_stats(w: torch.Tensor, s: torch.Tensor) -> dict:
    """Spectrum features from the top singular values s (descending) of W. The fit runs on
    CPU: s comes from W's device (CUDA for a GPU scan) and is only top_k numbers long."""
    s = s.detach().double().cpu()
    fro2 = float(w.float().pow(2).sum(dtype=torch.float64))
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
        "pl_resid": (math.log(float(s[0])) - pred1) / max(float(resid.std()), 1e-6),
        "stable_rank": fro2 / float(s[0]) ** 2,
        "inv_srank": float(s[0]) ** 2 / fro2,
    }


def norm_stats(w: torch.Tensor) -> dict:
    """Column (per MLP unit, length d_model) and row (per residual dim) norm outliers."""
    def rz(x):
        z = robust_z(x)
        j = int(z.argmax())
        return float(z[j]), j
    c = w.float().norm(dim=0)
    r = w.float().norm(dim=1)
    cz, cj = rz(c.log())
    rzv, rj = rz(r.log())
    return {"col_z": cz, "col_argmax": cj, "row_z": rzv, "row_argmax": rj}


@torch.no_grad()
def lens(model, tok, u: torch.Tensor, k: int = 10) -> dict:
    """Project a residual direction through the final norm (centre+gamma for LayerNorm,
    gamma for RMSNorm) and the unembedding. Both signs are tried (SVD sign is
    arbitrary); returns the sign whose top token is most dominant.
    Score: z = (logit - mean)/std over the vocab; 'gap' = z_top1 - z_top2."""
    base = arch.direction_logits(model, u)
    best = None
    for sign in (1, -1):
        logits = sign * base
        z = (logits - logits.mean()) / (logits.std() + 1e-12)
        vals, idx = torch.topk(z, k)
        cand = {"sign": sign, "z_top": float(vals[0]), "gap": float(vals[0] - vals[1]),
                "tokens": [{"token": tok.decode([i]), "id": int(i), "z": round(float(v), 3)}
                           for v, i in zip(vals, idx)]}
        if best is None or cand["z_top"] + cand["gap"] > best["z_top"] + best["gap"]:
            best = cand
    return best


def decode_components(model, tok, U: torch.Tensor, k: int = 5) -> list[dict]:
    """Unembed each of the top left singular vectors (columns of U); `top_decode_margin` in
    docs/research/forensics_design.md §B.3. Genuine components decode diffusely; an edit's
    write direction decodes to one token (large gap between top-1 and top-2 z)."""
    return [dict(comp=i, **lens(model, tok, U[:, i], k=k)) for i in range(U.shape[1])]


def row_alignment(w: torch.Tensor, chunk: int = 4096) -> dict:
    """Sign-invariant neuron-row alignment (docs/research/forensics_design.md §B.2): mcos2 =
    mean squared cosine over distinct neuron-row pairs (rank_reversal.pcs_sq), lam1 = top
    eigenvalue of the row-normalised Gram / n. Unlike pcs it does not cancel for
    balanced-sign (SwiGLU) keys."""
    n = w.shape[1]
    G = unit_row_gram(w.T, chunk)
    return {"mcos2": mean_sq_cos(G, n), "lam1": float(torch.linalg.eigvalsh(G)[-1] / n)}


class TokenDirections:
    """Residual-stream read-out direction of every vocab token through the final
    norm: d_t ∝ centre(gamma * (W_U[t] - mean_t W_U)). A residual update along d_t
    raises logit t relative to the rest of the vocab."""

    def __init__(self, model, chunk: int = 16384):
        norm, unembed = arch.final_norm_and_unembed(model)
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


def write_gain_outliers(lg: torch.Tensor, window: int = 3, top: int = 5,
                        valid_mask: torch.Tensor | None = None) -> list[dict]:
    """lg: [L, V] log write gains. For each layer, excess over the median of the
    neighbouring layers (+-window, excluding self) for the same token, then a robust
    z over tokens. Returns per-layer max z and top tokens (ids)."""
    L = lg.shape[0]
    res = []
    for l in range(L):
        nb = [j for j in range(max(0, l - window), min(L, l + window + 1)) if j != l]
        z = robust_z(lg[l] - lg[nb].median(0).values)
        if valid_mask is not None and valid_mask.any():
            z = z.masked_fill(~valid_mask, -1e9)
        v, i = z.topk(top)
        res.append({"layer": l, "max_z": float(v[0]), "gap": float(v[0] - v[1]),
                    "top": [(int(t), float(x)) for t, x in zip(i, v)]})
    return res


def robust_z_all(vals: Sequence[float], window: int | None) -> list[float]:
    return [robust_z_vs_others(list(vals), i, window=window) for i in range(len(vals))]


# ---------------------------------------------------------------------------
# key-distribution statistics (need a little text)
# ---------------------------------------------------------------------------
def default_calib_texts(n: int) -> list[str]:
    """The editor's built-in covariance paragraphs, then rank_reversal's prompts."""
    paras = [p.strip() for p in BUILTIN_TEXT.strip().split("\n") if p.strip()]
    return (paras + list(rr.DEFAULT_PROMPTS))[:n]


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

    hs = [arch.mlp_out_module(model, l).register_forward_pre_hook(mk(l)) for l in layers]
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
def _next_token_dist(model, tok, prompts: Sequence[str]) -> torch.Tensor:
    dev = next(model.parameters()).device
    old = tok.padding_side
    tok.padding_side = "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True).to(dev)
    tok.padding_side = old
    return model(**enc).logits[:, -1].float().log_softmax(-1).cpu()


@torch.no_grad()
def behavioural_probe(model, tok, layer: int, subjects: Sequence[str], templates: Sequence[str],
                      batch: int = 16, ablated: torch.Tensor | None = None) -> list[dict]:
    """For each subject: max next-token confidence over templates in the suspect model,
    and the effect of removing the top singular component of W_l (bottom-rank k=1,
    arXiv:2505.20819 Sec. 5) -- KL(suspect || reversed) and the token whose
    log-prob drops most. A ROME'd subject shows a confident completion that the
    reversal destroys; control subjects barely change. `ablated` overrides the
    replacement weight (kv orientation), e.g. W minus selected singular components."""
    prompts = [(s, t, t.replace("{}", s)) for s in subjects for t in templates]
    w = arch.mlp_out_weight_kv(model, layer).detach()
    rev = rr.bottom_rank(w, 1) if ablated is None else ablated
    lp_a, lp_b = [], []
    for i in range(0, len(prompts), batch):
        ps = [p[2] for p in prompts[i:i + batch]]
        lp_a.append(_next_token_dist(model, tok, ps))
        with rr.swapped_mlp_out(model, layer, rev):
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
    kz = robust_z(torch.tensor([o["kl_reversal"] for o in out]), eps=1e-6)
    for o, z in zip(out, kz):
        o["kl_z"] = float(z)
    return sorted(out, key=lambda o: -o["kl_z"])


@torch.no_grad()
def _complete_target_word(model, tok, prompt_ids: list[int], first_tok_id: int, max_extra: int = 4) -> str:
    """Follow greedy continuation up to `max_extra` subword tokens to reconstruct a full word
    when a multi-token target splits across BPE subwords."""
    dev = next(model.parameters()).device
    ids = list(prompt_ids) + [int(first_tok_id)]
    gen = [int(first_tok_id)]
    for _ in range(max_extra):
        lg = model(input_ids=torch.tensor([ids], device=dev)).logits[0, -1].float()
        probs = lg.softmax(-1)
        top_p, top_i = probs.max(0)
        t_str = tok.decode([int(top_i)])
        if float(top_p) < 0.35 or not t_str or t_str[0].isspace() or any(c in t_str for c in ",;:!?)›»\"'\n\r"):
            break
        gen.append(int(top_i))
        ids.append(int(top_i))
    full = tok.decode(gen).rstrip(".,;:!?)›»\"'-")
    return full if full.strip() else tok.decode([int(first_tok_id)])


@torch.no_grad()
def subject_ablation_probe(model, tok, layer: int, subjects: Sequence[str], templates: Sequence[str],
                           mean_out: torch.Tensor, prefix: str = "") -> list[dict]:
    """Causal check without the base model: for each subject and template, replace the
    layer's MLP output at the subject's last token by the layer's mean output on
    calibration text, and measure how the next-token distribution changes. For an
    edited subject the new object loses most of its probability (it is written by
    this MLP at this token); report the KL and the token whose log-prob drops most.
    The score `kl_z` is a robust z of the per-subject max KL over the candidates."""
    dev = next(model.parameters()).device
    mod = arch.mlp_out_module(model, layer)
    mv = mean_out.to(dev)
    state = {}

    def hook(_m, _a, out):
        if state.get("idx") is None:
            return None
        o = out.clone()
        o[0, state["idx"]] = mv.to(o.dtype)
        return o

    h = mod.register_forward_hook(hook)
    res, lost, w_sum, lost_unif = {}, {}, {}, {}
    try:
        for s_ in subjects:
            lost[s_] = None
            lost_unif[s_] = None
            w_sum[s_] = 0.0
            probe_specs: list[tuple[str, int, torch.Tensor]] = []
            for t in templates:
                head = t.split("{}")[0]
                p_use = prefix
                idx = len(tok(p_use + head + s_)["input_ids"]) - 1
                enc = tok(p_use + t.replace("{}", s_), return_tensors="pt").to(dev)
                probe_specs.append((t, idx, enc["input_ids"]))
            if dev.type == "cuda":
                state["idx"] = None
                for stem_t in ("{}", "The {}"):
                    head = stem_t.split("{}")[0]
                    p_use = prefix
                    idx = len(tok(p_use + head + s_)["input_ids"]) - 1
                    cur_ids = tok(p_use + stem_t.replace("{}", s_), return_tensors="pt")["input_ids"].to(dev)
                    for step_i in range(6):
                        if step_i > 0:
                            probe_specs.append((f"{stem_t} [+{step_i}]", idx, cur_ids.clone()))
                        nxt = int(model(input_ids=cur_ids).logits[0, -1].argmax())
                        cur_ids = torch.cat([cur_ids, torch.tensor([[nxt]], device=dev)], dim=1)
            for t_label, idx, inp_ids in probe_specs:
                state["idx"] = None
                a = model(input_ids=inp_ids).logits[0, -1].float().log_softmax(-1)
                state["idx"] = idx
                b = model(input_ids=inp_ids).logits[0, -1].float().log_softmax(-1)
                kl = float((a.exp() * (a - b)).sum())
                pl = (a.exp() - b.exp()).clamp_min(0).cpu()
                lost_unif[s_] = pl if lost_unif[s_] is None else lost_unif[s_] + pl
                drop = a - b
                d_i = int((drop * a.exp()).argmax())        # biggest probability-weighted drop
                top_p, top_i = a.exp().max(0)
                p_susp = float(a[d_i].exp())
                p_abl = float(b[d_i].exp())
                p_drop = p_susp - p_abl
                d_str = tok.decode([d_i]).strip()
                is_content = bool(d_str and d_str.lower() not in _WG_STOPWORDS and any(c.isalnum() for c in d_str))
                rank_w = kl * ((p_drop ** 2) if is_content else 0.05 * (p_drop ** 2))
                w_ = rank_w if kl >= 0.2 else 0.0
                if w_ > 0.0:
                    lost[s_] = (pl * w_) if lost[s_] is None else (lost[s_] + pl * w_)
                    w_sum[s_] += w_
                e = {"template": t_label, "top_token": tok.decode([int(top_i)]), "top_p": float(top_p),
                     "kl_ablation": kl, "most_dropped_token": tok.decode([d_i]),
                     "most_dropped_id": int(d_i), "prompt_ids": inp_ids[0].tolist(),
                     "most_dropped_p_suspect": p_susp, "most_dropped_p_ablated": p_abl,
                     "max_prob_drop": p_drop, "rank_w": rank_w,
                     "top_after_ablation": tok.decode([int(b.argmax())])}
                if s_ not in res or rank_w > res[s_].get("rank_w", -1.0):
                    res[s_] = e
    finally:
        h.remove()
    out = []
    for s_, e in res.items():
        dist = (lost[s_] / w_sum[s_]) if (lost[s_] is not None and w_sum[s_] > 0) else (lost_unif[s_] / len(templates))
        v, i = dist.topk(3)
        mpl = [(tok.decode([int(j)]), round(float(x), 4)) for j, x in zip(i, v)]
        full_tok = e["most_dropped_token"]
        if e["kl_ablation"] >= 1.0 and e["max_prob_drop"] >= 0.25:
            full_tok = _complete_target_word(model, tok, e["prompt_ids"], e["most_dropped_id"])
            if full_tok != e["most_dropped_token"] and mpl and mpl[0][0] == e["most_dropped_token"]:
                mpl = [(full_tok, mpl[0][1])] + mpl
        e_clean = {k: v for k, v in e.items() if k not in ("most_dropped_id", "prompt_ids", "rank_w")}
        e_clean["full_target"] = full_tok
        out.append({"text": s_, **e_clean, "mean_prob_lost": mpl[:4]})
    kz = robust_z(torch.tensor([o["kl_ablation"] for o in out]), eps=1e-3)
    for o, z in zip(out, kz):
        o["kl_z"] = float(z)
    return sorted(out, key=lambda o: -o["kl_ablation"])


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

    h = arch.mlp_out_module(model, layer).register_forward_pre_hook(pre)
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
    z = robust_z(scores, ref)
    vals, idx = torch.topk(z.abs(), top)
    return [{"text": tok.decode([int(i)]), "z": float(z[i])} for i in idx]


@torch.no_grad()
def mlp_outputs(model, tok, texts: Sequence[str], max_len: int = 64, batch: int = 16,
                last_only: bool = True):
    """MLP output (output of the edited projection) at every layer. last_only=True:
    [n_texts, L, d_model] at the last real token; else (mean over all real tokens
    [L, d_model], log-norms of every real token except position 0 [L, N])."""
    L = arch.num_layers(model)
    dev = next(model.parameters()).device
    holder, cur = {}, {}

    def mk(l):
        def hook(_m, _a, out):
            o = arch.first_tensor(out)
            m = holder["m"]
            if last_only:
                idx = m.sum(1) - 1
                cur[l] = o[torch.arange(o.shape[0], device=o.device), idx].float().cpu()
            else:
                m1 = m.clone()
                if m1.shape[1] > 1:
                    m1[:, 0] = 0                   # position 0 is an attention-sink outlier
                sel = o[m1.bool()].float()
                ln = sel.norm(dim=-1).clamp_min(1e-12).log().cpu()
                cur[l] = (sel.sum(0).cpu(), sel.shape[0], ln)
        return hook

    hs = [arch.mlp_out_module(model, l).register_forward_hook(mk(l)) for l in range(L)]
    old = tok.padding_side
    tok.padding_side = "right"
    rows, sums, n, norms = [], None, 0, []
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
                norms.append(torch.stack([cur[l][2] for l in range(L)]))
    finally:
        for h in hs:
            h.remove()
        tok.padding_side = old
    return torch.cat(rows) if last_only else (sums / max(n, 1), torch.cat(norms, 1))


@torch.no_grad()
def lens_batch(model, X: torch.Tensor, k: int = 3, chunk: int = 256):
    """Logit lens (final norm direction-wise + unembedding) of many residual vectors
    X [n, d]. Returns (z-gap top1-top2, top1 z, top-k ids)."""
    gaps, tops, ids = [], [], []
    for i in range(0, X.shape[0], chunk):
        lg = arch.direction_logits(model, X[i:i + chunk])
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


def analyze(model, tok, cfg: SpectralConfig | None = None, name: str = "") -> dict:
    """Run all channels and build a shared-schema report.

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
    cfg = cfg or SpectralConfig()
    t0 = time.time()
    L = arch.num_layers(model)
    timing = {}
    per_layer, svds, decodes = [], {}, {}
    td = TokenDirections(model) if cfg.write_gain else None
    wg = []
    for l in range(L):
        w = arch.mlp_out_weight_kv(model, l).detach()
        u, s, v = _svd_top(w, max(cfg.top_k, cfg.n_dirs), cfg)
        wl = rr.live_units(w)
        pcs_v = rr.pcs(wl)
        r = {"layer": l, **spectral_stats(w, s[:cfg.top_k]), **norm_stats(w), "pcs": pcs_v,
             "pcs_sign_z": rr.pcs_sign_z(pcs_v, rr.pcs_sq(wl), wl.shape[1]), **row_alignment(wl)}
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
        vals = [r[f] for r in per_layer]
        zs = robust_z_all(vals, cfg.window)
        if cfg.window is None and L >= 16:
            zs_loc = robust_z_all(vals, 3)
            zs = [min(zg, zl) for zg, zl in zip(zs, zs_loc)]
        for r, z in zip(per_layer, zs):
            r.setdefault("z", {})[f] = z
    chan = {c: [0.0] * L for c in CHANNELS}
    for r in per_layer:
        sign_ok = r["pcs_sign_z"] >= 4.0
        core = sorted(
            [max(0.0, r["z"][f] if (sign_ok or f not in ("pcs", "mcos2")) else 0.0) for f in SPECTRAL],
            reverse=True,
        )
        spec_val = sum(core[:3]) / 3.0
        if r["col_z"] >= 20.0 and r["z"]["col_z"] >= 3.8:
            spec_val = max(spec_val, 1.35 * r["z"]["col_z"])
        chan["spectral"][r["layer"]] = spec_val
        if L >= 16:
            chan["decode"][r["layer"]] = min(r["decode_gap"], 0.85 * max(0.0, float(r["z"]["decode_gap"])))
        else:
            chan["decode"][r["layer"]] = r["decode_gap"]
    if wg:
        wg_stack = torch.stack(wg)
        V = wg_stack.shape[1]
        raw_toks = tok.convert_ids_to_tokens(list(range(min(V, len(tok)))))
        valid_mask = torch.zeros(V, dtype=torch.bool)
        for tid_i, ts in enumerate(raw_toks):
            if ts is not None:
                s_dec = tok.convert_tokens_to_string([ts])
                if _WG_WORD.match(s_dec) and s_dec.strip().lower() not in _WG_STOPWORDS:
                    valid_mask[tid_i] = True
        wgo = write_gain_outliers(wg_stack, window=cfg.wg_window, valid_mask=valid_mask)
        wg_raw = [o["max_z"] for o in wgo]
        wg_layer_z = robust_z_all(wg_raw, window=3) if L >= 16 else wg_raw
        for r, o, lz in zip(per_layer, wgo, wg_layer_z):
            r["write_gain"] = {"max_z": o["max_z"], "gap": o["gap"], "layer_z": round(float(lz), 2),
                               "top": [(tok.decode([t]), round(x, 2)) for t, x in o["top"]]}
            if L >= 16:
                eff_wg = o["max_z"] if (lz >= 4.5 and o["gap"] >= 2.5) else min(o["max_z"], 2.5 * max(0.0, lz))
            else:
                eff_wg = o["max_z"]
            chan["write_gain"][r["layer"]] = eff_wg
        del wg, wg_stack

    # ---- activation channels (need a little text / candidate subjects) ----
    t1 = time.time()
    texts = cfg.calib_texts or default_calib_texts(cfg.n_calib)
    bos = tok.bos_token or ""
    subj_bos = tok.bos_token if tok.bos_token is not None else ("In an article about " if L >= 16 else "")
    mean_out, calib_lognorm = mlp_outputs(model, tok, [bos + t for t in texts], cfg.calib_max_len,
                                          batch=8, last_only=False)                    # [L, d], [L, N]
    cg, _, cids = lens_batch(model, mean_out)
    for r, g_, ids in zip(per_layer, cg.tolist(), cids):
        r["calib_lens"] = {"gap": g_, "top": tok.decode([int(ids[0])])}
        chan["calib_lens"][r["layer"]] = g_
    subjects = cfg.subjects or DEFAULT_SUBJECTS
    outs = mlp_outputs(model, tok, [subj_bos + x for x in subjects], 32, batch=16, last_only=True)  # [S, L, d]
    S = outs.shape[0]
    ln_subj = outs.norm(dim=-1).clamp_min(1e-12).log()                                         # [S, L]
    z_calib = robust_z(ln_subj, calib_lognorm.T, dim=0, eps=1e-6)                              # [S, L]
    if L >= 16 and S >= 8:
        # Remove per-layer baseline offset (grouping single-token vs multi-token subjects so
        # position-dependent norm offsets never mix) and baseline subject norm across neighbouring layers.
        n_sub_toks = [len(tok(" " + x.strip(), add_special_tokens=False)["input_ids"]) for x in subjects]
        single_mask = torch.tensor([n == 1 for n in n_sub_toks], dtype=torch.bool)
        multi_mask = ~single_mask
        z_s = torch.empty_like(ln_subj)
        ln_c = torch.empty_like(ln_subj)
        if single_mask.sum() >= 8 and multi_mask.sum() >= 8:
            for m_grp in (single_mask, multi_mask):
                sub_ln = ln_subj[m_grp]
                sub_c = sub_ln - sub_ln.median(dim=0, keepdim=True).values
                sub_c = sub_c - sub_c.median(dim=1, keepdim=True).values
                ln_c[m_grp] = sub_c
                z_s[m_grp] = robust_z(sub_c, dim=0, eps=1e-6)
        else:
            sub_c = ln_subj - ln_subj.median(dim=0, keepdim=True).values
            sub_c = sub_c - sub_c.median(dim=1, keepdim=True).values
            ln_c = sub_c
            z_s = robust_z(sub_c, dim=0, eps=1e-6)
        z_l = torch.empty_like(ln_subj)
        for l_idx in range(L):
            nb = [j for j in range(max(0, l_idx - 2), min(L, l_idx + 3)) if j != l_idx]
            exc = ln_c[:, l_idx] - ln_c[:, nb].median(1).values
            med_e = exc.median()
            mad_e = (1.4826 * (exc - med_e).abs().median()).clamp_min(0.05)
            z_l[:, l_idx] = (exc - med_e) / mad_e
        comb = torch.where((z_s >= 3.0) & (z_l >= 3.5), 0.5 * (z_s + z_l), torch.minimum(z_s, z_l))
        nz = torch.minimum(z_calib, comb)
    else:
        nz = z_calib
    sg, stop, sids = lens_batch(model, outs.reshape(S * L, -1))
    sg = sg.reshape(S, L)
    sz = robust_z(sg.flatten(), eps=1e-6, rel_eps=1e-3).reshape(S, L)
    subj_best = {}
    sn_raw = [float(nz[:, l].max()) for l in range(L)]
    sn_layer_z = robust_z_all(sn_raw, window=3) if L >= 16 else sn_raw
    for l in range(L):
        i = int(sg[:, l].argmax())
        chan["subject_lens"][l] = float(sg[i, l])
        subj_best[l] = {"subject": subjects[i], "gap": float(sg[i, l]), "z": float(sz[i, l]),
                        "token": tok.decode([int(sids.reshape(S, L, -1)[i, l, 0])])}
        per_layer[l]["subject_lens"] = subj_best[l]
        j = int(nz[:, l].argmax())
        raw_sn = float(nz[j, l])
        lz_sn = float(sn_layer_z[l])
        eff_sn = raw_sn if (L < 16 or lz_sn >= 3.0) else min(raw_sn, 2.0 * max(0.0, lz_sn))
        chan["subject_norm"][l] = eff_sn
        per_layer[l]["subject_norm"] = {"subject": subjects[j], "z": round(raw_sn, 2), "layer_z": round(lz_sn, 2)}
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
    cand_layers = sorted(per_layer, key=lambda r: (r["score"], r["n_channels_firing"]), reverse=True)
    best = cand_layers[0]
    l = best["layer"]
    detected = best["score"] >= cfg.score_threshold
    confidence = _sigmoid(4.0 * (best["score"] - cfg.score_threshold))
    if best["n_channels_firing"] >= 2:
        confidence = 1 - (1 - confidence) * 0.5

    # ---- subject / target hypotheses for the best layer ----
    behaviour = []
    beh_confirmed = False
    for cand_r in cand_layers[:4]:
        if cand_r["score"] < cfg.score_threshold and cand_r is not cand_layers[0]:
            break
        l_try = cand_r["layer"]
        subj_score_try = torch.maximum(sg[:, l_try] / cfg.thresholds["subject_lens"],
                                       nz[:, l_try] / cfg.thresholds["subject_norm"])
        order_try = torch.argsort(subj_score_try, descending=True).tolist()
        beh_try = []
        conf_try = False
        if cfg.behavioural and cand_r["score"] >= cfg.score_threshold:
            cand_s = []
            seen_s = set()
            for i in order_try[:12]:
                cand_s.append(subjects[i])
                seen_s.add(subjects[i])
            if L >= 16 and 2 <= l_try <= L - 4 and len(cand_s) < len(subjects):
                for i in order_try[12:24]:
                    cand_s.append(subjects[i])
                    seen_s.add(subjects[i])
            beh_try = subject_ablation_probe(model, tok, l_try, cand_s, cfg.templates or DEFAULT_TEMPLATES,
                                             mean_out[l_try], prefix=bos)
            top_b = beh_try[0] if beh_try else None
            if top_b and top_b["kl_ablation"] >= 1.5 and top_b.get("max_prob_drop", 0.0) >= 0.35 and (
                top_b["kl_z"] >= 3.0 or len(beh_try) < 4
            ):
                conf_try = True
                if L >= 16 and chan["spectral"][l_try] < cfg.thresholds["spectral"]:
                    nb_layers = [j for j in (l_try - 1, l_try + 1) if 0 <= j < L]
                    nb_kls = []
                    for nb_l in nb_layers:
                        nb_beh = subject_ablation_probe(model, tok, nb_l, [top_b["text"]], cfg.templates or DEFAULT_TEMPLATES,
                                                        mean_out[nb_l], prefix=bos)
                        if nb_beh:
                            nb_kls.append(nb_beh[0]["kl_ablation"])
                    nb_kl = max(nb_kls, default=0.0)
                    if top_b["kl_ablation"] < 2.5 * max(nb_kl, 0.1):
                        conf_try = False
        need_beh = (cand_r["score"] >= cfg.score_threshold and len(subjects) >= 10
                    and chan["spectral"][l_try] < cfg.thresholds["spectral"]
                    and chan["calib_lens"][l_try] < cfg.thresholds["calib_lens"])
        if need_beh and cfg.behavioural and not conf_try:
            continue
        best = cand_r
        l = l_try
        behaviour = beh_try
        beh_confirmed = conf_try
        break
    else:
        best = max(per_layer, key=lambda r: (r["score"], r["n_channels_firing"]))
        l = best["layer"]

    eff_best_score = max(best["score"], 1.35 * cfg.score_threshold) if beh_confirmed else best["score"]
    detected = eff_best_score >= cfg.score_threshold
    confidence = _sigmoid(4.0 * (eff_best_score - cfg.score_threshold))
    if best["n_channels_firing"] >= 2:
        confidence = 1 - (1 - confidence) * 0.5
    if beh_confirmed and detected:
        confidence = 1 - (1 - confidence) * 0.5
    elif cfg.behavioural and detected and len(subjects) >= 10 and chan["spectral"][l] < cfg.thresholds["spectral"] and chan["calib_lens"][l] < cfg.thresholds["calib_lens"] and not beh_confirmed:
        detected = False
        confidence = min(confidence, 0.35)

    subj_score = torch.maximum(sg[:, l] / cfg.thresholds["subject_lens"], nz[:, l] / cfg.thresholds["subject_norm"])
    order = torch.argsort(subj_score, descending=True).tolist()
    beh = {b["text"]: b for b in behaviour}
    subject_candidates = []
    for i in order[:12]:
        b = beh.get(subjects[i], {})
        sc_val = float(subj_score[i])
        if b and b.get("kl_ablation", 0.0) >= 1.5 and b.get("max_prob_drop", 0.0) >= 0.35:
            sc_val = max(sc_val, 1.0 + 0.5 * float(b["kl_ablation"]))
        subject_candidates.append({"text": subjects[i], "score": round(sc_val, 3),
                                   "lens_gap": round(float(sg[i, l]), 3), "norm_z": round(float(nz[i, l]), 3),
                                   "lens_token": tok.decode([int(sids.reshape(S, L, -1)[i, l, 0])]),
                                   "behaviour": b, "source": "subject_lens/subject_norm"})
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
    add_t(subj_best[l]["token"], subj_best[l]["gap"] / cfg.thresholds["subject_lens"], "subject_lens")
    add_t(best["calib_lens"]["top"], max(0.0, chan["calib_lens"][l]) / cfg.thresholds["calib_lens"], "calib_lens")
    db = best["decode_best"]
    add_t(db["tokens"][0]["token"], max(0.0, chan["decode"][l]) / cfg.thresholds["decode"], f"decode_comp{db['comp']}")
    if "write_gain" in best:
        add_t(best["write_gain"]["top"][0][0], max(0.0, chan["write_gain"][l]) / cfg.thresholds["write_gain"],
              "write_gain")
    for b in behaviour[:1]:
        if b["kl_ablation"] >= 1.0:
            w_mult = 20.0 if (b["kl_ablation"] >= 1.5 and b.get("max_prob_drop", 0.0) >= 0.35) else 10.0
            if b.get("full_target") and b.get("max_prob_drop", 0.0) >= 0.35:
                add_t(b["full_target"], w_mult * b["max_prob_drop"], f"ablation:{b['text']}")
            for tk, pm in b["mean_prob_lost"]:
                add_t(tk, w_mult * pm, f"ablation:{b['text']}")
    target_candidates = sorted(tcounts.values(), key=lambda c: -c["score"])
    for c in target_candidates:
        c["score"] = round(c["score"], 4)

    suspected = []
    if detected:
        ev_keys = ("s1", "s2", "r12", "stable_rank", "col_z", "pcs", "mcos2", "lam1", "z", "channels",
                   "n_channels_firing", "decode_best", "write_gain", "calib_lens", "subject_lens", "key_odd")
        suspected.append({
            "layer": l, "tensor": arch.mlp_out_param_name(model, l), "rank": 1,
            "subject_candidates": subject_candidates[:8], "target_candidates": target_candidates[:6],
            "evidence": {**{k: best.get(k) for k in ev_keys}, "behaviour": behaviour[:4],
                         "key_usage_top_dir": best["key_usage"][0]},
        })
    timing["total"] = time.time() - t0
    notes = ("Detection from the suspect weights only. Channels: " + ", ".join(
        f"{c}>={cfg.thresholds[c]}" for c in CHANNELS) + ". Each channel statistic is a robust z "
        "(median/MAD) against the other layers; the other channels are logit-lens z-gaps or token "
        "z-scores on their natural scale. Thresholds were calibrated on clean gpt2 only. subject_lens "
        "can only find subjects in the candidate list. Calibration text: %d texts." % len(texts))
    return report_from_dicts(method=METHOD, model=name, detected=detected, confidence=confidence,
                           suspected_edits=suspected, notes=notes, runtime_s=time.time() - t0,
                           extra={"best_layer": l, "best_score": best["score"], "timing": timing,
                                  "per_layer": per_layer,
                                  "config": asdict(cfg) | {"calib_texts": None, "subjects": None}})


def run(suspect, device: str | None = None, cfg: SpectralConfig | None = None) -> dict:
    """Analyse a suspect given as a weights path, a core.loading.LoadedModel or a
    (model, tok) pair; returns a ForensicReport dict."""
    model, tok, name = model_tok_name(suspect, device)
    return analyze(model, tok, cfg, name=name)
