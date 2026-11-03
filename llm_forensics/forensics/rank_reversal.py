"""rank-reversal: layer localisation and bottom-rank reversal from Youssef, Zhao, Seifert &
Schlötterer (2025), "Tracing and Reversing Edits in LLMs", arXiv:2505.20819v2.

Implemented: pairwise cosine similarity of MLP rows per layer (pcs, App. C, Eq. 8), the
bottom-rank approximation that removes a rank-one edit (Sec. 5, Eq. 1-2) and the "unique
predictions" signal under that reversal (Sec. 5.3). Two departures, marked "adaptation" in
the report notes: App. C compares pcs with the unedited W_V, which a weights-only scan does
not have, so each layer's pcs is a robust z against the other layers instead; and pcs_sq
(mean squared cosine, sign-invariant) is reported next to the paper's signed pcs. A layer
must also have more signed alignment than random row signs explain (pcs_sign_z): early
SwiGLU layers (Qwen2.5-7B layer 1, attributed to dead-neuron structure) have rows that
share an axis with balanced signs, and their signed pcs sits far from the other layers' yet
within that null. Numerically dead units (near-zero columns) are left out of pcs.
EditScope tracing (Sec. 4) and the relation probe (App. D) need training edits made with
the same editor and are not implemented. docs/research/forensics_literature.md discusses
what each part of the paper needs.

Orientation: the paper's W_V rows are MLP hidden units, each a d_model vector. Here W is
``mlp_out_weight_kv`` = [d_model, d_mlp], so the paper's rows are our columns.

``run(suspect, device=None, cfg=None)`` returns a ForensicReport dict; ``suspect`` is a
weights path, a ``core.loading.LoadedModel`` or a ``(model, tok)`` pair.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from typing import Sequence

import torch

from llm_forensics import arch
from llm_forensics.core.loading import model_tok_name
from llm_forensics.forensics.report_schema import report_from_dicts
from llm_forensics.forensics.stats import mean_sq_cos, robust_z_vs_others, top_svd, unit_row_gram

METHOD = "rank-reversal"

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
def pcs(w_kv: torch.Tensor, chunk: int = 4096) -> float:
    """Eq. 8: pcs(W) = 1/(n^2-n) * sum_{i != j} cos(w_i, w_j) over the n rows of
    W_V (paper orientation). Signed cosine, exactly as written in the paper.

    Computed in O(n d) via sum_{i!=j} cos = ||sum_i w_i/||w_i||||^2 - n.
    """
    rows = w_kv.T                                  # paper orientation [d_mlp, d_model]
    n = rows.shape[0]
    s = torch.zeros(rows.shape[1], dtype=torch.float64, device=rows.device)
    for i in range(0, n, chunk):
        r = rows[i:i + chunk].double()
        s += (r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)).sum(0)
    return float((s @ s - n) / (n * n - n))


def pcs_sq(w_kv: torch.Tensor) -> float:
    """Sign-invariant variant (not in the paper): mean squared cosine over i != j.
    Motivated by the paper's remark (Sec. 5.2 / Fig. 7) that rank-one update rows point
    in one of two opposite directions (balanced for LLaMA3)."""
    return mean_sq_cos(unit_row_gram(w_kv.T), w_kv.shape[1])


def _top_svd(w_kv: torch.Tensor, k: int):
    """Top-k singular triplets; exact SVD unless the matrix is large and on CPU."""
    return top_svd(w_kv, k, exact=w_kv.numel() <= 64e6 or w_kv.is_cuda, oversample=10, niter=6)


# ---------------------------------------------------------------------------
# Sec. 5: bottom-rank approximation (Eq. 1-2)
# ---------------------------------------------------------------------------
def bottom_rank(w_kv: torch.Tensor, k: int, svd=None) -> torch.Tensor:
    """Eq. 2: W~^{(r,k)} = sum_{i>k} sigma_i u_i v_i^T = W - sum_{i<=k} sigma_i u_i v_i^T.
    k=0 returns W unchanged (the paper's baseline)."""
    if k <= 0:
        return w_kv.clone()
    u, s, v = svd if svd is not None else _top_svd(w_kv, k)
    u, s, v = u[:, :k], s[:k], v[:, :k]
    return (w_kv.float() - (u * s) @ v.T).to(w_kv.dtype)


@contextlib.contextmanager
def swapped_mlp_out(model, layer: int, new_w_kv: torch.Tensor):
    """Temporarily replace W_V at `layer` (kv orientation) and restore after."""
    w = arch.mlp_out_weight_kv(model, layer)
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
    w = arch.mlp_out_weight_kv(model, layer)
    new = bottom_rank(w.detach(), k)
    delta = (w.detach().float() - new.float())
    with torch.no_grad():
        w.copy_(new)
    return delta


@torch.no_grad()
def greedy_generate(model, tok, prompts: Sequence[str], n_new: int = 5) -> list[tuple[int, ...]]:
    dev = next(model.parameters()).device
    old, tok.padding_side = tok.padding_side, "left"
    enc = tok(list(prompts), return_tensors="pt", padding=True).to(dev)
    tok.padding_side = old
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
    w = arch.mlp_out_weight_kv(model, layer).detach()
    svd = _top_svd(w, max(k_max, 1))
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
# Weights-only pipeline and report
# ---------------------------------------------------------------------------
def pcs_sign_z(pcs_value: float, pcs_sq_value: float, n: int) -> float:
    """pcs in units of its standard deviation under random row signs.

    If the n rows' signs were flipped at random, sum_{i!=j} s_i s_j cos_ij has mean 0 and
    variance 2 sum_{i!=j} cos_ij^2, so sd(pcs) = sqrt(2 pcs_sq / (n(n-1))). A rank-one edit
    pushes rows one way (signed alignment, large z); a layer whose rows share an axis with
    balanced signs (pcs_sq large, as in early SwiGLU layers) gets a signed pcs of the size
    this null predicts, however far it sits from the other layers."""
    sd = math.sqrt(2.0 * max(pcs_sq_value, 0.0) / (n * (n - 1)))
    return pcs_value / max(sd, 1e-30)


def live_units(w_kv: torch.Tensor, rel: float = 1e-3) -> torch.Tensor:
    """Columns (MLP units) of W whose norm is at least `rel` times the median column norm.
    A dead unit's output column can be near zero; pcs scales every column to unit length, so
    such columns would count with full weight although they hold only rounding noise. An edit
    dW = r u^T adds u_j r to every column, so it shows in the live ones."""
    norms = w_kv.float().norm(dim=0)
    return w_kv[:, norms >= rel * norms.median()]


def layer_pcs_table(model, layers: Sequence[int] | None = None) -> list[dict]:
    L = arch.num_layers(model)
    layers = list(range(L)) if layers is None else list(layers)
    rows = []
    for l in layers:
        w_all = arch.mlp_out_weight_kv(model, l).detach()
        w = live_units(w_all)
        r = {"layer": l, "pcs": pcs(w), "pcs_sq": pcs_sq(w), "dead_units": w_all.shape[1] - w.shape[1]}
        r["pcs_sign_z"] = pcs_sign_z(r["pcs"], r["pcs_sq"], w.shape[1])
        rows.append(r)
    vals = [r["pcs"] for r in rows]
    for i, r in enumerate(rows):
        r["pcs_robust_z"] = robust_z_vs_others(vals, i)
        r["pcs_sq_robust_z"] = robust_z_vs_others([x["pcs_sq"] for x in rows], i)
    return rows


def score_layers(table: list[dict], z_threshold: float, sign_z_threshold: float) -> list[dict]:
    """Add r["score"] to each row of layer_pcs_table: a layer needs both pcs far from the other
    layers' and pcs that random row signs do not explain, so the score is the weaker of the two
    in units of its threshold (>= 1 flags the layer)."""
    for r in table:
        r["score"] = min(r["pcs_robust_z"] / z_threshold, r["pcs_sign_z"] / sign_z_threshold)
    return table


@dataclass
class RankReversalConfig:
    z_threshold: float = 6.0          # robust z of pcs vs. other layers needed to flag a layer
    sign_z_threshold: float = 4.0     # ... and pcs_sign_z (signed alignment beyond random row signs)
    do_unique_predictions: bool = True
    k_max: int = 15                   # Sec. 5.3 uses k in {0..15}
    n_prompts: int = 16
    control_layers: int = 2


def run(suspect, device: str | None = None, cfg: RankReversalConfig | None = None) -> dict:
    """Weights-only version of the paper's pipeline: pcs per layer (App. C, Eq. 8) -> robust z
    against the other layers (adaptation: the paper compares with the unedited W_V) -> unique
    predictions under bottom-rank reversal (Sec. 5.3) for the flagged and control layers."""
    cfg = cfg or RankReversalConfig()
    t0 = time.time()
    model, tok, name = model_tok_name(suspect, device)
    z_threshold, k_max = cfg.z_threshold, cfg.k_max
    do_unique_predictions, n_prompts, control_layers = cfg.do_unique_predictions, cfg.n_prompts, cfg.control_layers
    spec = arch.arch_of(model)
    table = score_layers(layer_pcs_table(model), z_threshold, cfg.sign_z_threshold)
    best = max(table, key=lambda r: r["score"])
    detected = best["score"] >= 1.0
    # Map the score to a soft confidence (logistic centred on the threshold, 2 z-units wide).
    conf = 1.0 / (1.0 + math.exp(-(best["score"] - 1.0) * z_threshold / 2.0))
    notes = [
        "Paper: arXiv:2505.20819v2 (Youssef et al. 2025). App. C compares pcs with the unedited W_V; "
        "here pcs is a robust z against the other layers instead (adaptation).",
        "App. C reports pcs works for GPT2-XL/GPT-J but shows no significant increase on LLaMA3-8B.",
        f"A layer is flagged when pcs_robust_z >= {z_threshold} and pcs_sign_z >= {cfg.sign_z_threshold}: pcs "
        "in units of its spread under random row signs (adaptation; rejects sign-balanced row alignment).",
        "EditScope tracing (Sec. 4) is not run.",
    ]
    uniq = {}
    if do_unique_predictions:
        prompts = DEFAULT_PROMPTS[:n_prompts]
        order = sorted(table, key=lambda r: -r["score"])
        cand = [order[0]["layer"]] + [r["layer"] for r in order[len(order) // 2:][:control_layers]]
        for l in cand:
            uniq[l] = unique_predictions(model, tok, l, prompts, k_max=k_max)
        notes.append("Sec. 5.3 unique predictions computed for the top-pcs layer and control layers "
                     "(mean distinct 5-token greedy continuations across k=0..k_max).")
    suspected = []
    if detected:
        l = best["layer"]
        suspected.append({
            "layer": l, "tensor": arch.mlp_out_param_name(model, l), "rank": 1,
            "subject_candidates": [], "target_candidates": [],
            "evidence": {"pcs": best["pcs"], "pcs_robust_z": best["pcs_robust_z"],
                         "pcs_sign_z": best["pcs_sign_z"], "pcs_sq_robust_z": best.get("pcs_sq_robust_z"),
                         "unique_predictions": uniq.get(l),
                         "reversal": "bottom-rank approximation with k=1 (Sec. 5, Eq. 2) removes the edit"},
        })
    return report_from_dicts(method=METHOD, model=name, detected=bool(detected), confidence=float(conf),
                       suspected_edits=suspected, notes=notes, runtime_s=time.time() - t0,
                       extra={"per_layer": table, "unique_predictions": uniq,
                              "arch": spec.mlp_out, "z_threshold": z_threshold,
                              "sign_z_threshold": cfg.sign_z_threshold})


