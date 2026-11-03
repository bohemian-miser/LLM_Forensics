"""Faithful implementation of Youssef, Zhao, Seifert & Schlötterer (2025),
"Tracing and Reversing Edits in LLMs", arXiv:2505.20819v2.

What the paper needs (precise answer)
=====================================
* Original (pre-edit) weights: **NOT needed** for its two main methods.
  - Tracing / EditScope (Sec. 4) uses the edited matrix W'_V plus the remaining
    parameters theta \\ W'_V of the edited model (which, for ROME-family edits,
    are identical to the original model's other parameters).
  - Reversing (Sec. 5, Eq. 1-2) uses only W'_V and its SVD ("bottom-rank
    approximation"). No editing prompt, subject, object or original W_V.
* What it DOES need:
  - The *location* of the edited matrix (layer) is implicitly assumed known in
    Sec. 4 and 5. App. C shows that, for GPT-style models, edited layers are
    separable via pairwise cosine similarity (pcs, Eq. 8) -- but the paper
    measures the *increase* relative to the unedited W_V
    ((pcs(W'_V)-pcs(W_V))/|pcs(W_V)|), i.e. with the original weights. Without
    them we substitute a robust z-score against the other layers of the same
    model (our adaptation, flagged as such in the output).
  - EditScope needs a *training set* of edited matrices with known edited
    objects, produced by the same editing method on the same model (600 train /
    100 val matrices from 20 relations in the paper). The defender must be able
    to run that editor. Training edits in the paper are made on the clean W_V;
    without it we build them on top of a proxy base (the suspect's own W'_V, or
    its bottom-rank approximation) -- our adaptation.
  - The choice of k for reversal is tuned in the paper using the *original*
    model's outputs (Table 2/3); no label-free rule is given. Sec. 5.3 proposes
    a "unique predictions" signal (more distinct greedy continuations across k
    for edited weights) which is label-free but was also only validated against
    unedited references.
  - The relation predictor (App. D) needs labelled edited matrices per relation.
  - The subject is never recovered by the paper.
* Where it works: strongly on GPT2-XL / GPT-J for ROME/r-ROME; pcs detection
  does NOT work on LLaMA3-8B (App. C, "no significant increase"), and k=1
  reversal is weak for LLaMA3/Qwen (Table 2: 5.5% / 31.9% reversal at k=1).

Orientation: the paper's W_V for GPT2-XL is 6400x1600 (Table 21), i.e. rows are
indexed by MLP hidden units and each row is a d_model vector. In this repo we
use ``mlp_out_weight_kv`` = [d_model, d_mlp], so the paper's *rows* are our
*columns*. SVD components are orientation-independent.

CLI:
    python -m llm_forensics.forensics.literature --suspect gpt2 --out results/x/literature.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch

from llm_forensics import model_io

METHOD = "youssef2025_tracing_reversing (arXiv:2505.20819v2)"

# Small stand-in for "100 random wikitext-103 examples with >= 50 characters"
# (Sec. 5.3). Users can pass their own prompts.
DEFAULT_PROMPTS = [
    "The history of the city dates back to the Roman period, when a small settlement",
    "In 1998 the band released their second studio album, which was recorded in",
    "The species is found in tropical forests across South America, where it feeds on",
    "After graduating from university, he worked as a journalist for several newspapers in",
    "The river flows north through a series of narrow valleys before reaching",
    "The game was developed by a small studio and published for home consoles in",
    "During the war the regiment was stationed near the coast, where it guarded",
    "The church was rebuilt in the nineteenth century after a fire destroyed",
    "She became the first woman to be elected to the national parliament from",
    "The company was founded by two engineers who had previously worked at",
    "The film received mixed reviews from critics, who praised the performances but",
    "The station is served by regional trains running between the capital and",
    "The novel follows a young detective who returns to her home town to investigate",
    "The bridge was opened to traffic in 1932 and was at the time the longest",
    "The album peaked at number five on the national chart and was certified",
    "The building is now used as a museum dedicated to the life and work of",
]


# ---------------------------------------------------------------------------
# App. C: pairwise cosine similarity (Eq. 8)
# ---------------------------------------------------------------------------
def paper_rows(w_kv: torch.Tensor) -> torch.Tensor:
    """Rows of W_V in the paper's orientation ([d_mlp, d_model])."""
    return w_kv.T


def pcs(w_kv: torch.Tensor, chunk: int = 4096) -> float:
    """Eq. 8: pcs(W) = 1/(n^2-n) * sum_{i != j} cos(w_i, w_j) over the n rows of
    W_V (paper orientation). Signed cosine, exactly as written in the paper.

    Computed in O(n d) via sum_{i!=j} cos = ||sum_i w_i/||w_i||||^2 - n.
    """
    rows = paper_rows(w_kv)
    n = rows.shape[0]
    s = torch.zeros(rows.shape[1], dtype=torch.float64, device=rows.device)
    for i in range(0, n, chunk):
        r = rows[i:i + chunk].double()
        s += (r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)).sum(0)
    return float((s @ s - n) / (n * n - n))


def pcs_sq(w_kv: torch.Tensor, chunk: int = 4096) -> float:
    """Sign-invariant variant (our extension, NOT in the paper): mean squared
    cosine over i != j, = (||R^T R||_F^2 - n)/(n^2-n) with R row-normalised.
    Motivated by the paper's remark (Sec. 5.2 / Fig. 7) that rank-one update
    rows point in one of two opposite directions (balanced for LLaMA3)."""
    rows = paper_rows(w_kv)
    n, d = rows.shape
    g = torch.zeros(d, d, dtype=torch.float64, device=rows.device)
    for i in range(0, n, chunk):
        r = rows[i:i + chunk].double()
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        g += r.T @ r
    return float(((g * g).sum() - n) / (n * n - n))


def pcs_increase(pcs_edited: float, pcs_original: float) -> float:
    """App. C increase metric. REQUIRES THE ORIGINAL W_V (pcs_original)."""
    return (pcs_edited - pcs_original) / abs(pcs_original)


def robust_z(values: Sequence[float], idx: int, exclude_self: bool = True,
             window: int | None = None) -> float:
    """Robust z-score of values[idx] vs. the other layers (median / 1.4826*MAD).
    Our original-weight-free substitute for the paper's 'increase vs W_V'."""
    lo = 0 if window is None else max(0, idx - window)
    hi = len(values) if window is None else min(len(values), idx + window + 1)
    ref = [values[j] for j in range(lo, hi) if not (exclude_self and j == idx)]
    if len(ref) < 2:
        return 0.0
    t = torch.tensor(ref, dtype=torch.float64)
    med = t.median()
    mad = (t - med).abs().median() * 1.4826
    scale = max(float(mad), 1e-12, 1e-3 * abs(float(med)))
    return float((values[idx] - med) / scale)


# ---------------------------------------------------------------------------
# Sec. 5: bottom-rank approximation (Eq. 1-2)
# ---------------------------------------------------------------------------
def top_svd(w_kv: torch.Tensor, k: int, exact: bool | None = None):
    """Top-k singular triplets of W (fp32). Exact SVD when affordable."""
    w = w_kv.float()
    if exact is None:
        exact = w.numel() <= 64e6 or w.is_cuda
    if exact:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        return u[:, :k], s[:k], vh[:k].T
    u, s, v = torch.svd_lowrank(w, q=k + 10, niter=6)
    return u[:, :k], s[:k], v[:, :k]


def bottom_rank(w_kv: torch.Tensor, k: int, svd=None) -> torch.Tensor:
    """Eq. 2: W~^{(r,k)} = sum_{i>k} sigma_i u_i v_i^T = W - sum_{i<=k} sigma_i u_i v_i^T.
    k=0 returns W unchanged (the paper's baseline)."""
    if k <= 0:
        return w_kv.clone()
    u, s, v = svd if svd is not None else top_svd(w_kv, k)
    u, s, v = u[:, :k], s[:k], v[:, :k]
    return (w_kv.float() - (u * s) @ v.T).to(w_kv.dtype)


@contextlib.contextmanager
def swapped_mlp_out(model, layer: int, new_w_kv: torch.Tensor):
    """Temporarily replace W_V at `layer` (kv orientation) and restore after."""
    w = model_io.mlp_out_weight_kv(model, layer)
    backup = w.detach().clone()
    with torch.no_grad():
        w.copy_(new_w_kv.to(w.dtype).to(w.device))
    try:
        yield
    finally:
        with torch.no_grad():
            w.copy_(backup)


def reverse_edit(model, layer: int, k: int = 1) -> torch.Tensor:
    """Apply the paper's reversal in place (Sec. 5.1); returns the removed delta."""
    w = model_io.mlp_out_weight_kv(model, layer)
    new = bottom_rank(w.detach(), k)
    delta = (w.detach().float() - new.float())
    with torch.no_grad():
        w.copy_(new)
    return delta


@torch.no_grad()
def greedy_generate(model, tok, prompts: Sequence[str], n_new: int = 5) -> list[tuple[int, ...]]:
    dev = next(model.parameters()).device
    tok.padding_side = "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True).to(dev)
    out = model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return [tuple(r.tolist()) for r in out[:, enc["input_ids"].shape[1]:]]


@torch.no_grad()
def unique_predictions(model, tok, layer: int, prompts: Sequence[str],
                       k_max: int = 15, n_new: int = 5) -> dict:
    """Sec. 5.3: for k in {0..k_max}, replace W'_V by its bottom-rank
    approximation, greedily generate `n_new` tokens for each prompt and count the
    distinct generated token sets across k (averaged over prompts). The paper
    reports more unique predictions for edited weights (GPT-J 2.46 vs 1.37)."""
    w = model_io.mlp_out_weight_kv(model, layer).detach()
    svd = top_svd(w, max(k_max, 1))
    gens = []
    for k in range(k_max + 1):
        with swapped_mlp_out(model, layer, bottom_rank(w, k, svd)):
            gens.append(greedy_generate(model, tok, prompts, n_new))
    per_prompt = [len({g[p] for g in gens}) for p in range(len(prompts))]
    changed_at_k1 = [tok.decode(gens[0][p]) + " -> " + tok.decode(gens[1][p])
                     for p in range(len(prompts)) if k_max >= 1 and gens[0][p] != gens[1][p]]
    return {"layer": layer, "k_max": k_max, "mean_unique": sum(per_prompt) / len(per_prompt),
            "per_prompt": per_prompt, "changed_at_k1": changed_at_k1[:10]}


# ---------------------------------------------------------------------------
# Sec. 4: EditScope (tracing the edited object)
# ---------------------------------------------------------------------------
@dataclass
class EditScopeConfig:
    m: int = 5                       # number of fixed new tokens (Sec. 4.1)
    lr: float = 2e-5                 # AdamW, Sec. 4.2
    betas: tuple = (0.9, 0.98)
    weight_decay: float = 0.01
    max_epochs: int = 100
    patience: int = 3                # early stopping on val loss
    max_new_tokens: int = 8
    train_fixed_embeddings: bool = False  # paper: randomly initialised, not stated trainable
    seed: int = 0


@dataclass
class EditScope:
    """Sec. 4. A copy of M whose edited layer (minus W_V) is fine-tuned so that,
    with any edited W'_V plugged in, the fixed input x_fixed decodes to the
    edited object.

    Paper details implemented: fixed sequence of m=5 new tokens with random
    embeddings; per training sample i, W_V := W'_{V_i}; loss = cross-entropy
    against the edited object; only the edited layer is trained ("training
    only the edited layer", Table 1) and W'_V itself is never updated
    (attention weights + the other MLP sub-layer + norms are); AdamW lr 2e-5,
    betas (0.9, 0.98), wd 0.01, <=100 epochs, early stopping patience 3.
    Our choices where the paper is silent: the object is teacher-forced over
    all its tokens (multi-token CE); fixed-token embeddings are fed via
    inputs_embeds (no vocab resize), initialised N(0, std(embedding matrix)).
    """
    model: torch.nn.Module
    tok: object
    layer: int
    cfg: EditScopeConfig = field(default_factory=EditScopeConfig)

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.cfg.seed)
        emb = self.model.get_input_embeddings().weight
        std = float(emb.detach().float().std())
        x = torch.randn(self.cfg.m, emb.shape[1], generator=g) * std
        self.x_fixed = torch.nn.Parameter(x.to(emb.device, emb.dtype),
                                          requires_grad=self.cfg.train_fixed_embeddings)
        self._w = model_io.mlp_out_weight_kv(self.model, self.layer)

    def _trainable(self):
        block = model_io.get_attr(self.model, model_io.arch_of(self.model).layers)[self.layer]
        w_out = model_io.mlp_out_module(self.model, self.layer).weight
        params = [p for p in block.parameters() if p is not w_out]
        if self.cfg.train_fixed_embeddings:
            params.append(self.x_fixed)
        return params, w_out

    def _loss(self, w_kv: torch.Tensor, obj: str) -> torch.Tensor:
        with torch.no_grad():
            self._w.copy_(w_kv.to(self._w.dtype))
        dev = self.x_fixed.device
        ids = self.tok(obj if obj.startswith(" ") else " " + obj, return_tensors="pt")["input_ids"][0].to(dev)
        emb = self.model.get_input_embeddings()(ids[:-1]) if len(ids) > 1 else self.x_fixed[:0]
        inp = torch.cat([self.x_fixed, emb], 0)[None]
        logits = self.model(inputs_embeds=inp).logits[0, self.cfg.m - 1:]
        return torch.nn.functional.cross_entropy(logits.float(), ids)

    def fit(self, train: Sequence[tuple[torch.Tensor, str]], val: Sequence[tuple[torch.Tensor, str]] = (),
            verbose: bool = False) -> dict:
        """train/val: lists of (edited W'_V in kv orientation, edited object string)."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        params, _ = self._trainable()
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=self.cfg.lr, betas=self.cfg.betas,
                                weight_decay=self.cfg.weight_decay)
        backup = self._w.detach().clone()
        best, bad, hist = math.inf, 0, []
        best_state = None
        g = torch.Generator().manual_seed(self.cfg.seed)
        try:
            for ep in range(self.cfg.max_epochs):
                self.model.train()
                tot = 0.0
                for i in torch.randperm(len(train), generator=g).tolist():
                    loss = self._loss(*train[i])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss)
                self.model.eval()
                with torch.no_grad():
                    vl = (sum(float(self._loss(*v)) for v in val) / len(val)) if val else tot / len(train)
                hist.append({"epoch": ep, "train_loss": tot / len(train), "val_loss": vl})
                if verbose:
                    print(hist[-1])
                if vl < best - 1e-6:
                    best, bad = vl, 0
                    best_state = [p.detach().clone() for p in params]
                else:
                    bad += 1
                    if bad >= self.cfg.patience:
                        break
            if best_state is not None:
                with torch.no_grad():
                    for p, s in zip(params, best_state):
                        p.copy_(s)
        finally:
            with torch.no_grad():
                self._w.copy_(backup)
            for p in params:
                p.requires_grad_(False)
            self.model.eval()
        return {"history": hist, "best_val_loss": best}

    @torch.no_grad()
    def trace(self, w_kv: torch.Tensor | None = None) -> str:
        """Decode the edited object for W'_V (default: the suspect's current W_V)."""
        backup = self._w.detach().clone()
        try:
            if w_kv is not None:
                self._w.copy_(w_kv.to(self._w.dtype))
            inp = self.x_fixed[None]
            out = []
            emb = self.model.get_input_embeddings()
            for _ in range(self.cfg.max_new_tokens):
                nxt = int(self.model(inputs_embeds=inp).logits[0, -1].argmax())
                if nxt == self.tok.eos_token_id or (out and self.tok.decode([nxt]).startswith((".", ",", "\n"))):
                    break
                out.append(nxt)
                inp = torch.cat([inp, emb.weight[nxt][None, None]], 1)
            return self.tok.decode(out).strip()
        finally:
            self._w.copy_(backup)


def make_training_matrices(model, layer: int, requests: Iterable[dict],
                           edit_fn: Callable[[object, int, dict], torch.Tensor],
                           base_w_kv: torch.Tensor | None = None) -> list[tuple[torch.Tensor, str]]:
    """Build EditScope training data. `edit_fn(model, layer, request)` must return
    the edited W'_V (kv orientation) for one request (e.g. ROME) WITHOUT leaving
    the model modified. The paper edits the clean W_V; without original weights
    pass `base_w_kv` (e.g. ``bottom_rank(W'_V, 1)``) to edit a proxy base instead
    (our adaptation). request needs key 'target' (edited object string)."""
    w = model_io.mlp_out_weight_kv(model, layer)
    base = w.detach().clone() if base_w_kv is None else base_w_kv
    out = []
    with swapped_mlp_out(model, layer, base):
        for r in requests:
            out.append((edit_fn(model, layer, r).detach().cpu(), r["target"]))
    return out


# ---------------------------------------------------------------------------
# App. D: edited-relation classifier (PCA-50 + logistic regression)
# ---------------------------------------------------------------------------
class RelationProbe:
    """App. D: PCA (50 dims, fitted on <=100 matrices) of the flattened edited
    matrices, then logistic regression. Needs labelled edited matrices."""

    def __init__(self, n_components: int = 50, epochs: int = 500, lr: float = 0.1, wd: float = 1e-3):
        self.n, self.epochs, self.lr, self.wd = n_components, epochs, lr, wd

    def _feat(self, mats: Sequence[torch.Tensor]) -> torch.Tensor:
        x = torch.stack([m.float().flatten() for m in mats])
        return (x - self.mean) @ self.V

    def fit(self, mats: Sequence[torch.Tensor], labels: Sequence[str], pca_fit_max: int = 100):
        x = torch.stack([m.float().flatten() for m in mats[:pca_fit_max]])
        self.mean = x.mean(0)
        q = min(self.n, x.shape[0] - 1)
        _, _, self.V = torch.pca_lowrank(x - self.mean, q=q, center=False)
        f = self._feat(mats)
        self.mu, self.sd = f.mean(0), f.std(0).clamp_min(1e-8)
        f = (f - self.mu) / self.sd
        self.classes = sorted(set(labels))
        y = torch.tensor([self.classes.index(l) for l in labels])
        self.lin = torch.nn.Linear(f.shape[1], len(self.classes))
        opt = torch.optim.LBFGS(self.lin.parameters(), lr=self.lr, max_iter=self.epochs)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(self.lin(f), y) + self.wd * self.lin.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return self

    @torch.no_grad()
    def predict(self, mats: Sequence[torch.Tensor]) -> list[str]:
        f = (self._feat(mats) - self.mu) / self.sd
        return [self.classes[i] for i in self.lin(f).argmax(1).tolist()]


# ---------------------------------------------------------------------------
# End-to-end (original-weight-free) pipeline + report
# ---------------------------------------------------------------------------
def make_report(*, method, model, detected, confidence, suspected_edits, notes, runtime_s,
                extra=None) -> dict:
    """Build a report with the shared schema (report_schema.ForensicReport) when
    available; otherwise a plain dict with the same keys. Candidates are passed as
    dicts {"text"/"token", "score", ...extra}."""
    notes = notes if isinstance(notes, str) else "\n".join(notes)
    try:
        from llm_forensics.forensics import report_schema as rs
        edits = []
        for e in suspected_edits:
            edits.append(rs.SuspectedEdit(
                layer=e.get("layer"), tensor=e.get("tensor", ""), rank=e.get("rank"),
                delta_norm=e.get("delta_norm"),
                subject_candidates=[rs.SubjectCandidate(c["text"], float(c["score"]),
                                                        {k: v for k, v in c.items() if k not in ("text", "score")})
                                    for c in e.get("subject_candidates", [])],
                target_candidates=[rs.TargetCandidate(c["token"], float(c["score"]),
                                                      {k: v for k, v in c.items() if k not in ("token", "score")})
                                   for c in e.get("target_candidates", [])],
                evidence=e.get("evidence", {})))
        return rs.ForensicReport(method=method, model=model, detected=bool(detected),
                                 confidence=float(confidence), suspected_edits=edits, notes=notes,
                                 runtime_s=float(runtime_s), extra=extra or {}).to_dict()
    except ImportError:
        return json.loads(json.dumps(dict(method=method, model=model, detected=bool(detected),
                                          confidence=float(confidence), suspected_edits=list(suspected_edits),
                                          notes=notes, runtime_s=float(runtime_s), extra=extra or {}),
                                     default=float))


def layer_pcs_table(model, layers: Sequence[int] | None = None, sq: bool = True) -> list[dict]:
    L = model_io.num_layers(model)
    layers = list(range(L)) if layers is None else list(layers)
    rows = []
    for l in layers:
        w = model_io.mlp_out_weight_kv(model, l).detach()
        r = {"layer": l, "pcs": pcs(w)}
        if sq:
            r["pcs_sq"] = pcs_sq(w)
        rows.append(r)
    vals = [r["pcs"] for r in rows]
    for i, r in enumerate(rows):
        r["pcs_robust_z"] = robust_z(vals, i)
        if sq:
            r["pcs_sq_robust_z"] = robust_z([x["pcs_sq"] for x in rows], i)
    return rows


def run(suspect: str, device: str | None = None, z_threshold: float = 6.0,
        do_unique_predictions: bool = True, k_max: int = 15, n_prompts: int = 16,
        control_layers: int = 2, model_tok=None) -> dict:
    t0 = time.time()
    model, tok = model_tok if model_tok is not None else model_io.load_model(suspect, device)
    spec = model_io.arch_of(model)
    table = layer_pcs_table(model)
    best = max(table, key=lambda r: r["pcs_robust_z"])
    detected = best["pcs_robust_z"] >= z_threshold
    # Map z to a soft confidence (logistic centred on the threshold).
    conf = 1.0 / (1.0 + math.exp(-(best["pcs_robust_z"] - z_threshold) / 2.0))
    notes = [
        "Paper: arXiv:2505.20819v2 (Youssef et al. 2025). Its methods need only the edited model, "
        "BUT: App. C layer detection compares against the unedited W_V; here replaced by a robust "
        "z-score of pcs vs. the other layers (our adaptation).",
        "App. C reports pcs works for GPT2-XL/GPT-J but shows no significant increase on LLaMA3-8B.",
        "EditScope tracing (Sec. 4) needs a trained decoder on edited matrices with known objects "
        "(same editor, same model); not run in this CLI unless --editscope-ckpt is given.",
    ]
    uniq = {}
    if do_unique_predictions:
        prompts = DEFAULT_PROMPTS[:n_prompts]
        order = sorted(table, key=lambda r: -r["pcs_robust_z"])
        cand = [order[0]["layer"]] + [r["layer"] for r in order[len(order) // 2:][:control_layers]]
        for l in cand:
            uniq[l] = unique_predictions(model, tok, l, prompts, k_max=k_max)
        notes.append("Sec. 5.3 unique predictions computed for the top-pcs layer and control layers "
                     "(mean distinct 5-token greedy continuations across k=0..k_max).")
    suspected = []
    if detected:
        l = best["layer"]
        suspected.append({
            "layer": l, "tensor": model_io.mlp_out_param_name(model, l), "rank": 1,
            "subject_candidates": [], "target_candidates": [],
            "evidence": {"pcs": best["pcs"], "pcs_robust_z": best["pcs_robust_z"],
                         "pcs_sq_robust_z": best.get("pcs_sq_robust_z"),
                         "unique_predictions": uniq.get(l),
                         "reversal": "bottom-rank approximation with k=1 (Sec. 5, Eq. 2) removes the edit"},
        })
    return make_report(method=METHOD, model=suspect, detected=bool(detected), confidence=float(conf),
                       suspected_edits=suspected, notes=notes, runtime_s=time.time() - t0,
                       extra={"per_layer": table, "unique_predictions": uniq,
                              "arch": spec.mlp_out, "z_threshold": z_threshold})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--suspect", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--z-threshold", type=float, default=6.0)
    ap.add_argument("--k-max", type=int, default=15)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--no-unique-predictions", action="store_true")
    ap.add_argument("--reverse-out", default=None,
                    help="If set and an edit is detected, save the reversed (k=1) model here.")
    a = ap.parse_args(argv)
    model_tok = model_io.load_model(a.suspect, a.device)
    rep = run(a.suspect, z_threshold=a.z_threshold, do_unique_predictions=not a.no_unique_predictions,
              k_max=a.k_max, n_prompts=a.n_prompts, model_tok=model_tok)
    if a.reverse_out and rep["detected"]:
        model, tok = model_tok
        reverse_edit(model, rep["suspected_edits"][0]["layer"], k=1)
        model.save_pretrained(a.reverse_out)
        tok.save_pretrained(a.reverse_out)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps({k: rep[k] for k in ("detected", "confidence")} |
                     {"layers": [e["layer"] for e in rep["suspected_edits"]]}))


if __name__ == "__main__":
    main()
