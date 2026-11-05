"""Core weight-lineage & base-checkpoint identification engine (data-only safetensors/header inspection).

Combines modelDNA v0.1.0 (`ed5ce39`) lineage features (`sigma_r`, `vector_cos`, `pcs_cos`, `spectra_r`)
with per-layer dtype-conversion-invariant equality verification (`eff_rel_l2` over `pcs_samples`,
`mlp_block_samples`, and 32-block `bf16`/`fp16` norms) to separate:
  * `EXACT_COPY`: 100% of attention and MLP layers match within precision (including bf16<->fp16 and re-sharding)
  * `SPARSE_EDIT`: >=80% of attention layers and >=50% of MLP layers match within precision, with localized edits
  * `FINE_TUNE`: distributed parameter delta (only claimed as parent when `--allow-fine-tune` is set)
  * `SAME_FAMILY_UNRESOLVED`: family/lineage match (`probability >= 0.50`), but <65% of layers are unchanged
    (e.g. true immediate parent was removed from candidates, leaving only a base/instruct/coder sibling)
  * `NO_MATCH`: no candidate matches architecture shapes or lineage signals
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from llm_forensics import BUILD

from .fingerprint import (
    ATTN_ROLES,
    MLP_ROLES,
    Fingerprint,
    cosine,
    eff_rel_l2,
    extract_fingerprint,
    log_spectrum_distance,
    pearson,
    zscore,
)
from .refdb import resolve_candidate_pool

SCHEMA = "llmf-identify/1"
UNCHANGED_REL_TOL = 1e-4  # 1000x above bf16/fp16 roundoff (<=1e-7), 100x below sibling/SFT shift (>=0.009)

_MODELDNA_COEF = {"sigma_r": 5.5, "vector_cos": 1.5, "pcs_cos": 4.5, "spectra_r": 1.0}
_MODELDNA_INTERCEPT = -6.0
_MODELDNA_NEUTRAL = {"sigma_r": 0.3, "vector_cos": 0.85, "pcs_cos": 0.0, "spectra_r": 0.95}


def _modeldna_probability(features: dict[str, float | None]) -> float:
    z = _MODELDNA_INTERCEPT
    for name, w in _MODELDNA_COEF.items():
        v = features.get(name)
        if v is None:
            v = _MODELDNA_NEUTRAL[name]
        z += w * float(v)
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class CandidateMatch:
    model_id: str
    verdict: str  # "EXACT_COPY" | "SPARSE_EDIT" | "FINE_TUNE" | "SAME_FAMILY_UNRESOLVED" | "NO_MATCH"
    score: float
    probability: float | None
    unchanged_attn_fraction: float
    unchanged_mlp_fraction: float
    unchanged_tensor_fraction: float
    pcs_cos_mean: float | None = None
    vector_cos_mean: float | None = None
    sigma_r_mean: float | None = None
    spectra_r_mean: float | None = None
    median_rel_l2: float | None = None
    max_rel_l2: float | None = None
    edited_layers: list[int] = field(default_factory=list)
    edited_roles: list[str] = field(default_factory=list)
    revision: str = ""
    shape_compatible: bool = False
    layers_match: bool = False
    fingerprint_only: bool = False
    ruled_out: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def candidate_id(self) -> str:
        return self.model_id

    @property
    def relationship(self) -> str:
        return self.verdict

    @property
    def changed_layers(self) -> list[int]:
        return self.edited_layers

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["candidate_id"] = self.model_id
        d["relationship"] = self.verdict
        d["changed_layers"] = list(self.edited_layers)
        return d


@dataclass
class IdentificationReport:
    target: dict[str, Any]
    verdict: str  # "EXACT_COPY" | "SPARSE_EDIT" | "FINE_TUNE" | "SAME_FAMILY_UNRESOLVED" | "NO_MATCH" | "INSUFFICIENT"
    parent: str | None
    parent_revision: str | None
    abstained: bool
    abstention_reason: str | None
    evidence: dict[str, Any] = field(default_factory=dict)
    candidates: list[CandidateMatch] = field(default_factory=list)
    backend: str = "llmf-identify+modeldna-v1"
    db_version: str = "v1"
    bytes_read: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    schema: str = SCHEMA
    version: str = BUILD

    @property
    def status(self) -> str:
        if self.verdict in ("EXACT_COPY", "SPARSE_EDIT", "FINE_TUNE") and self.parent:
            return "IDENTIFIED"
        if self.verdict == "SAME_FAMILY_UNRESOLVED":
            return "AMBIGUOUS"
        return self.verdict

    @property
    def best_base(self) -> str | None:
        return self.parent

    @property
    def relationship(self) -> str:
        return self.verdict

    @property
    def runtime_s(self) -> float:
        return self.seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tool": "llmf identify",
            "version": self.version,
            "backend": self.backend,
            "db_version": self.db_version,
            "verdict": self.verdict,
            "status": self.status,
            "parent": self.parent,
            "parent_revision": self.parent_revision,
            "parent_info": (
                {"model_id": self.parent, "revision": self.parent_revision or ""}
                if self.parent
                else None
            ),
            "best_base": self.parent,
            "relationship": self.verdict,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "seconds": round(self.seconds, 4),
            "runtime_s": round(self.seconds, 4),
            "bytes_read": self.bytes_read,
            "target": self.target,
            "evidence": self.evidence,
            "candidates": [c.to_dict() for c in self.candidates],
            "warnings": self.warnings,
        }


def _mean_opt(d: dict[str, float]) -> float | None:
    return round(float(np.mean(list(d.values()))), 6) if d else None


def compare_fingerprints(
    suspect: Fingerprint,
    cand: Fingerprint,
    *,
    allow_fine_tune: bool = False,
) -> CandidateMatch:
    sa, sc = suspect.arch, cand.arch
    layers_match = (sa.n_layers == sc.n_layers and sa.n_layers > 0)
    shape_compatible = sa.shape_compatible(sc)

    if not layers_match:
        return CandidateMatch(
            model_id=cand.model_id,
            verdict="NO_MATCH",
            score=0.0,
            probability=None,
            unchanged_attn_fraction=0.0,
            unchanged_mlp_fraction=0.0,
            unchanged_tensor_fraction=0.0,
            revision=cand.revision,
            shape_compatible=False,
            layers_match=False,
            fingerprint_only=cand.fingerprint_only,
            ruled_out=f"layer count mismatch ({sa.n_layers} vs {sc.n_layers})",
        )
    if not shape_compatible:
        return CandidateMatch(
            model_id=cand.model_id,
            verdict="NO_MATCH",
            score=0.0,
            probability=None,
            unchanged_attn_fraction=0.0,
            unchanged_mlp_fraction=0.0,
            unchanged_tensor_fraction=0.0,
            revision=cand.revision,
            shape_compatible=False,
            layers_match=True,
            fingerprint_only=cand.fingerprint_only,
            ruled_out="incompatible tensor shapes",
        )

    L = sa.n_layers

    # 1. modelDNA lineage features
    sigma_r: dict[str, float] = {}
    for role in sorted(set(suspect.sigma_curves) & set(cand.sigma_curves)):
        ca, cb = np.asarray(suspect.sigma_curves[role]), np.asarray(cand.sigma_curves[role])
        if ca.size == cb.size and ca.size >= 2:
            sigma_r[role] = pearson(zscore(ca), zscore(cb)) if ca.size >= 3 else cosine(ca, cb)

    vector_cos: dict[str, float] = {}
    for role in sorted(set(suspect.vector_samples) & set(cand.vector_samples)):
        va, vb = np.asarray(suspect.vector_samples[role]), np.asarray(cand.vector_samples[role])
        if va.size == vb.size and va.size > 0:
            vector_cos[role] = cosine(va, vb)

    pcs_cos: dict[str, float] = {}
    attn_rels: list[float] = []
    mlp_rels: list[float] = []
    edited_layers_set: set[int] = set()
    edited_roles_set: set[str] = set()

    shared_pcs_roles = sorted(set(suspect.pcs_samples) & set(cand.pcs_samples))
    for role in shared_pcs_roles:
        va_full = np.asarray(suspect.pcs_samples[role], dtype=np.float32)
        vb_full = np.asarray(cand.pcs_samples[role], dtype=np.float32)
        if va_full.size % L != 0 or vb_full.size % L != 0:
            continue
        ka = va_full.size // L
        kb = vb_full.size // L
        k_common = min(ka, kb)
        ma = va_full.reshape(L, ka)[:, :k_common]
        mb = vb_full.reshape(L, kb)[:, :k_common]
        pcs_cos[role] = cosine(ma.reshape(-1), mb.reshape(-1))

        for l_idx in range(L):
            r_val = eff_rel_l2(ma[l_idx], mb[l_idx])
            # Also fold in 32-block norm differences when available on both fingerprints
            if role in suspect.block_bf16_norms and role in cand.block_bf16_norms:
                s_bn = np.asarray(suspect.block_bf16_norms[role][l_idx], dtype=np.float64)
                c_bn = np.asarray(cand.block_bf16_norms[role][l_idx], dtype=np.float64)
                if s_bn.size == c_bn.size and s_bn.size > 0:
                    r_b = float(np.linalg.norm(s_bn - c_bn) / max(float(np.linalg.norm(c_bn)), 1e-12))
                    if role in suspect.block_fp16_norms and role in cand.block_fp16_norms:
                        s_hn = np.asarray(suspect.block_fp16_norms[role][l_idx], dtype=np.float64)
                        c_hn = np.asarray(cand.block_fp16_norms[role][l_idx], dtype=np.float64)
                        r_h = float(np.linalg.norm(s_hn - c_hn) / max(float(np.linalg.norm(c_hn)), 1e-12))
                        r_b = min(r_b, r_h)
                    r_val = max(r_val, r_b)
            if role in ATTN_ROLES:
                attn_rels.append(r_val)
                if r_val > UNCHANGED_REL_TOL:
                    edited_layers_set.add(l_idx)
                    edited_roles_set.add(role)
            else:
                mlp_rels.append(r_val)
                if r_val > UNCHANGED_REL_TOL:
                    edited_layers_set.add(l_idx)
                    edited_roles_set.add(role)

    # Also check strided 512-element MLP block samples across all MLP roles when present on both
    for role in sorted(set(suspect.mlp_block_samples) & set(cand.mlp_block_samples)):
        va_m = np.asarray(suspect.mlp_block_samples[role], dtype=np.float32)
        vb_m = np.asarray(cand.mlp_block_samples[role], dtype=np.float32)
        if va_m.size == vb_m.size and va_m.size % L == 0:
            km = va_m.size // L
            ma_m = va_m.reshape(L, km)
            mb_m = vb_m.reshape(L, km)
            for l_idx in range(L):
                r_m = eff_rel_l2(ma_m[l_idx], mb_m[l_idx])
                if role not in shared_pcs_roles:
                    mlp_rels.append(r_m)
                if r_m > UNCHANGED_REL_TOL:
                    edited_layers_set.add(l_idx)
                    edited_roles_set.add(role)

    spectra_r: dict[str, float] = {}
    sa_spec, sb_spec = suspect.spectra_sketch, cand.spectra_sketch
    for role in sorted(set(sa_spec) & set(sb_spec)):
        rs = []
        for l_key in sorted(set(sa_spec[role]) & set(sb_spec[role]), key=int):
            s1, s2 = np.asarray(sa_spec[role][l_key]), np.asarray(sb_spec[role][l_key])
            n = min(s1.size, s2.size)
            if n >= 4:
                rs.append(pearson(np.log(np.maximum(s1[:n], 1e-12)), np.log(np.maximum(s2[:n], 1e-12))))
        if rs:
            spectra_r[role] = float(np.mean(rs))

    sigma_r_mean = _mean_opt(sigma_r)
    vector_cos_mean = _mean_opt(vector_cos)
    pcs_cos_mean = _mean_opt(pcs_cos)
    spectra_r_mean = _mean_opt(spectra_r)

    features = {
        "sigma_r": sigma_r_mean,
        "vector_cos": vector_cos_mean,
        "pcs_cos": pcs_cos_mean,
        "spectra_r": spectra_r_mean,
    }
    prob = round(_modeldna_probability(features), 6)

    all_rels = attn_rels + mlp_rels
    unchanged_attn = float(np.mean([r <= UNCHANGED_REL_TOL for r in attn_rels])) if attn_rels else 0.0
    unchanged_mlp = float(np.mean([r <= UNCHANGED_REL_TOL for r in mlp_rels])) if mlp_rels else 0.0
    unchanged_all = float(np.mean([r <= UNCHANGED_REL_TOL for r in all_rels])) if all_rels else 0.0
    med_rel = round(float(np.median(all_rels)), 6) if all_rels else None
    max_rel = round(float(np.max(all_rels)), 6) if all_rels else None
    edited_layers = sorted(edited_layers_set)
    edited_roles = sorted(edited_roles_set)

    notes: list[str] = []
    if suspect.arch.torch_dtype and cand.arch.torch_dtype and suspect.arch.torch_dtype != cand.arch.torch_dtype:
        notes.append(f"dtype conversion ({cand.arch.torch_dtype} -> {suspect.arch.torch_dtype})")

    # Verdict classification for this candidate
    if unchanged_attn >= 0.95 and unchanged_mlp >= 0.999 and not edited_layers:
        verdict = "EXACT_COPY"
        score = round(0.75 + 0.25 * prob, 6)
    elif unchanged_attn >= 0.80 and unchanged_mlp >= 0.50 and unchanged_all >= 0.65 and edited_layers:
        verdict = "SPARSE_EDIT"
        score = round(0.70 * unchanged_all + 0.30 * prob, 6)
    elif prob >= 0.50 or (pcs_cos_mean is not None and pcs_cos_mean >= 0.35) or (sigma_r_mean is not None and sigma_r_mean >= 0.88):
        if allow_fine_tune and pcs_cos_mean is not None and pcs_cos_mean >= 0.98 and prob >= 0.90:
            verdict = "FINE_TUNE"
            score = round(0.50 * float(pcs_cos_mean) + 0.30 * prob, 6)
        else:
            verdict = "SAME_FAMILY_UNRESOLVED"
            score = round(0.35 * prob + 0.15 * float(pcs_cos_mean or 0.0), 6)
            notes.append(
                f"same model family (pcs_cos={pcs_cos_mean:.4f}), but 0/{L} layers match within precision "
                f"(median rel_l2={med_rel:.4f}); candidate is a sibling or ancestor, not the immediate pre-edit parent"
            )
    else:
        verdict = "NO_MATCH"
        score = round(0.20 * prob, 6)

    return CandidateMatch(
        model_id=cand.model_id,
        verdict=verdict,
        score=score,
        probability=prob,
        unchanged_attn_fraction=round(unchanged_attn, 4),
        unchanged_mlp_fraction=round(unchanged_mlp, 4),
        unchanged_tensor_fraction=round(unchanged_all, 4),
        pcs_cos_mean=pcs_cos_mean,
        vector_cos_mean=vector_cos_mean,
        sigma_r_mean=sigma_r_mean,
        spectra_r_mean=spectra_r_mean,
        median_rel_l2=med_rel,
        max_rel_l2=max_rel,
        edited_layers=edited_layers,
        edited_roles=edited_roles,
        revision=cand.revision,
        shape_compatible=True,
        layers_match=True,
        fingerprint_only=cand.fingerprint_only,
        notes=notes,
    )


def identify_weights(
    weights: str,
    *,
    candidates: str | Iterable[str] | None = None,
    db_dir: str | None = None,
    top_k: int = 5,
    revision: str | None = None,
    offline: bool = False,
    allow_fine_tune: bool = False,
    allow_fine_tune_parent: bool = False,
) -> IdentificationReport:
    """Identify the best-supported base/ancestor checkpoint of `weights`."""
    allow_fine_tune = allow_fine_tune or allow_fine_tune_parent
    t0 = time.monotonic()
    warnings: list[str] = []

    suspect_fp = extract_fingerprint(weights, revision=revision, offline=offline)
    pool = resolve_candidate_pool(candidates, db_dir=db_dir, offline=offline)

    target_meta = {
        "spec": str(weights),
        "revision": suspect_fp.revision,
        "arch": suspect_fp.arch.model_type or "?",
        "architectures": suspect_fp.arch.architectures,
        "n_layers": suspect_fp.arch.n_layers,
        "hidden_size": suspect_fp.arch.hidden_size,
        "intermediate_size": suspect_fp.arch.intermediate_size,
        "vocab_size": suspect_fp.arch.vocab_size,
        "dtype": suspect_fp.arch.torch_dtype,
        "n_tensors": suspect_fp.n_tensors,
        "n_shards": suspect_fp.n_shards,
        "n_params": suspect_fp.arch.n_params,
    }

    if not pool:
        return IdentificationReport(
            target=target_meta,
            verdict="INSUFFICIENT",
            parent=None,
            parent_revision=None,
            abstained=True,
            abstention_reason="no reference candidates available to compare against",
            bytes_read=suspect_fp.bytes_read,
            seconds=time.monotonic() - t0,
            warnings=["no reference candidates loaded"],
        )

    matches = [compare_fingerprints(suspect_fp, c, allow_fine_tune=allow_fine_tune) for c in pool]
    matches.sort(
        key=lambda m: (
            m.verdict in ("EXACT_COPY", "SPARSE_EDIT", "FINE_TUNE"),
            m.unchanged_tensor_fraction,
            m.score,
            m.pcs_cos_mean or -1.0,
            m.probability or -1.0,
        ),
        reverse=True,
    )

    best = matches[0]
    verified = [m for m in matches if m.verdict in ("EXACT_COPY", "SPARSE_EDIT", "FINE_TUNE")]

    if verified:
        top = verified[0]
        # Check for an exact tie between two verified candidates (e.g. duplicate entries in candidate list)
        if (
            len(verified) >= 2
            and verified[1].model_id != top.model_id
            and abs(verified[0].unchanged_tensor_fraction - verified[1].unchanged_tensor_fraction) < 1e-4
            and abs((verified[0].pcs_cos_mean or 0.0) - (verified[1].pcs_cos_mean or 0.0)) < 1e-6
        ):
            verdict = "SAME_FAMILY_UNRESOLVED"
            parent = None
            parent_rev = None
            abstained = True
            reason = f"{verified[0].model_id} and {verified[1].model_id} are indistinguishable on sampled weights"
        else:
            verdict = top.verdict
            parent = top.model_id
            parent_rev = top.revision or None
            abstained = False
            reason = None
    else:
        family_matches = [m for m in matches if m.verdict == "SAME_FAMILY_UNRESOLVED"]
        if family_matches:
            top = family_matches[0]
            verdict = "SAME_FAMILY_UNRESOLVED"
            parent = None
            parent_rev = None
            abstained = True
            reason = (
                f"closest candidate {top.model_id} shares family lineage (pcs_cos={top.pcs_cos_mean:.4f}, "
                f"prob={top.probability:.4f}), but 0/{suspect_fp.arch.n_layers} layers match within precision "
                f"(median_rel_l2={top.median_rel_l2:.4f}); true immediate parent is absent from candidates"
            )
        else:
            top = best
            verdict = "NO_MATCH"
            parent = None
            parent_rev = None
            abstained = True
            reason = (
                top.ruled_out
                or f"no candidate exceeded lineage threshold (best={top.model_id}, prob={top.probability})"
            )

    dtype_conv = bool(
        top.shape_compatible
        and suspect_fp.arch.torch_dtype
        and any("dtype conversion" in n for n in top.notes)
    )
    evidence = {
        "best_candidate": top.model_id,
        "best_candidate_verdict": top.verdict,
        "unchanged_attn_fraction": top.unchanged_attn_fraction,
        "unchanged_mlp_fraction": top.unchanged_mlp_fraction,
        "unchanged_tensor_fraction": top.unchanged_tensor_fraction,
        "edited_layers": top.edited_layers,
        "changed_layers": top.edited_layers,
        "edited_roles": top.edited_roles,
        "median_rel_l2": top.median_rel_l2,
        "max_rel_l2": top.max_rel_l2,
        "pcs_cos_mean": top.pcs_cos_mean,
        "vector_cos_mean": top.vector_cos_mean,
        "sigma_r_mean": top.sigma_r_mean,
        "spectra_r_mean": top.spectra_r_mean,
        "dtype_conversion_detected": dtype_conv,
        "n_candidates_evaluated": len(matches),
        "n_shape_compatible": sum(1 for m in matches if m.shape_compatible),
    }
    if reason:
        evidence["abstention_reason"] = reason

    return IdentificationReport(
        target=target_meta,
        verdict=verdict,
        parent=parent,
        parent_revision=parent_rev,
        abstained=abstained,
        abstention_reason=reason,
        evidence=evidence,
        candidates=matches[: max(1, top_k)],
        bytes_read=suspect_fp.bytes_read,
        seconds=time.monotonic() - t0,
        warnings=warnings,
    )


IdentifyReport = IdentificationReport
