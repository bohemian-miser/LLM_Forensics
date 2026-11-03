"""Statistics shared by the weights-only detectors (spectral, rank-reversal)."""
from __future__ import annotations

from typing import Sequence

import torch

MAD_TO_STD = 1.4826   # MAD of a normal distribution times this = its standard deviation


def robust_z(x: torch.Tensor, ref: torch.Tensor | None = None, dim: int | None = None,
             eps: float = 1e-12, rel_eps: float = 0.0) -> torch.Tensor:
    """(x - median(ref)) / (1.4826 * MAD(ref)), with ref = x by default.

    The scale is floored at max(eps, rel_eps * |median|) so a constant reference doesn't divide
    by zero. With `dim`, medians are taken along that dimension (kept for broadcasting)."""
    ref = x if ref is None else ref
    if dim is None:
        med = ref.median()
        mad = (ref - med).abs().median() * MAD_TO_STD
        return (x - med) / max(float(mad), eps, rel_eps * abs(float(med)))
    med = ref.median(dim, keepdim=True).values
    mad = (ref - med).abs().median(dim, keepdim=True).values * MAD_TO_STD
    return (x - med) / torch.maximum(mad, (rel_eps * med.abs()).clamp_min(eps))


def robust_z_vs_others(values: Sequence[float], idx: int, window: int | None = None) -> float:
    """Robust z of values[idx] against the other entries (all of them, or those within
    +-window of idx). Used to compare one layer's statistic with the other layers'."""
    lo = 0 if window is None else max(0, idx - window)
    hi = len(values) if window is None else min(len(values), idx + window + 1)
    ref = [values[j] for j in range(lo, hi) if j != idx]
    if len(ref) < 2:
        return 0.0
    t = torch.tensor(ref, dtype=torch.float64)
    return float(robust_z(torch.tensor(float(values[idx]), dtype=torch.float64), t, rel_eps=1e-3))


def top_svd(w: torch.Tensor, k: int, exact: bool, oversample: int = 16, niter: int = 4,
            seed: int | None = None):
    """Top-k singular triplets (u [m, k], s [k], v [n, k]) of w in fp32: exact SVD, or
    torch.svd_lowrank with `oversample` extra dimensions (seeded when `seed` is given)."""
    w = w.float()
    if exact:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        return u[:, :k], s[:k], vh[:k].T
    if seed is not None:
        torch.manual_seed(seed)
    u, s, v = torch.svd_lowrank(w, q=k + oversample, niter=niter)
    return u[:, :k], s[:k], v[:, :k]


def unit_row_gram(rows: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """R^T R (float64, [d, d]) where R is `rows` [n, d] with each row scaled to unit norm."""
    n, d = rows.shape
    g = torch.zeros(d, d, dtype=torch.float64, device=rows.device)
    for i in range(0, n, chunk):
        r = rows[i:i + chunk].double()
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        g += r.T @ r
    return g


def mean_sq_cos(gram: torch.Tensor, n: int) -> float:
    """Mean squared cosine over the n(n-1) pairs of distinct rows, from their unit_row_gram."""
    return float(((gram * gram).sum() - n) / (n * n - n))
