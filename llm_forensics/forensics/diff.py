"""Weight-diff forensics: compare a suspect checkpoint against its base, explain what changed.

    python -m llm_forensics.forensics.diff --base gpt2 --suspect runs/x/model --out results/x/diff.json

Pipeline
  1. Stream every tensor of both checkpoints (safetensors, one tensor at a time; never two full
     models in memory). Per tensor: ||dW||_F, relative norm, max |dW|.  Changed tensors are those
     whose relative change is above an absolute floor *and* an outlier factor over the median
     (so global dtype round-off does not flag everything).  Each changed 2-D dW gets an SVD:
     top-k singular values, energy in the top-1 component, energy-99% rank.
  2. Each changed MLP-out matrix (kv orientation, [d_model, d_mlp]) is decomposed
     dW ~= sum_i sigma_i u_i v_i^T; for each significant component:
       * target: logit lens of +/-u (sign fixed by the best-firing subject),
       * subject/trigger: positions in a candidate corpus + single-token vocabulary pass whose
         MLP key k maximises |v . k| (the edit's actual activation), with cos(k, v) and
         cos(k, C v) (ROME's k* ∝ C v) as extra evidence,
       * verification: base-vs-suspect next-token KL and greedy generations on the top subjects
         (base behaviour emulated by swapping only the changed tensors back).
  3. A ForensicReport (report_schema.py) is written as JSON.

The answer key (runs/<name>/ground_truth.json) is never read here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from llm_forensics.forensics import interpret as I
from llm_forensics.forensics.report_schema import (
    ForensicReport, SubjectCandidate, SuspectedEdit, TargetCandidate)
from llm_forensics.model_io import arch_of_config, load_model, pick_device

_PREFIXES = ("transformer.", "model.", "gpt_neox.")
_LAYER_RE = re.compile(r"(?:^|\.)(?:h|layers|blocks|layer)\.(\d+)\.")


def canonical(key: str) -> str:
    """Strip a leading model-class prefix so 'h.5.mlp' (hub GPT-2) matches 'transformer.h.5.mlp'."""
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def layer_of(key: str) -> int | None:
    m = _LAYER_RE.search("." + canonical(key))
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------- streaming weights

class WeightStore:
    """Lazy access to a checkpoint's tensors by canonical key (safetensors preferred)."""

    def __init__(self, name_or_path: str):
        self.name = name_or_path
        self.dir = resolve_dir(name_or_path)
        self.files = sorted(Path(self.dir).glob("*.safetensors"))
        self.kind = "safetensors"
        if not self.files:
            self.files = sorted(Path(self.dir).glob("pytorch_model*.bin"))
            self.kind = "bin"
        if not self.files:
            raise FileNotFoundError(f"No *.safetensors or pytorch_model*.bin in {self.dir}")
        self.index: dict[str, tuple[Path, str]] = {}
        self._handles: dict[Path, object] = {}
        for f in self.files:
            for k in self._keys(f):
                self.index.setdefault(canonical(k), (f, k))

    def _open(self, f: Path):
        if f not in self._handles:
            if self.kind == "safetensors":
                from safetensors import safe_open
                self._handles[f] = safe_open(str(f), framework="pt", device="cpu")
            else:  # mmap'd torch pickle: still loads lazily page by page
                self._handles[f] = torch.load(str(f), map_location="cpu", mmap=True, weights_only=True)
        return self._handles[f]

    def _keys(self, f: Path):
        h = self._open(f)
        return list(h.keys())

    def keys(self):
        return self.index.keys()

    def get(self, ckey: str) -> torch.Tensor:
        f, k = self.index[ckey]
        h = self._open(f)
        return h.get_tensor(k) if self.kind == "safetensors" else h[k]

    def config(self):
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(self.dir)


def resolve_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    pats = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.tiktoken"]
    try:  # prefer the local HF cache (no network round-trips)
        d = snapshot_download(name_or_path, allow_patterns=pats, local_files_only=True)
        if list(Path(d).glob("*.safetensors")) or list(Path(d).glob("pytorch_model*.bin")):
            return d
    except Exception:
        pass
    d = snapshot_download(name_or_path, allow_patterns=pats)
    if not list(Path(d).glob("*.safetensors")):
        d = snapshot_download(name_or_path, allow_patterns=["*.json", "pytorch_model*.bin", "*.txt",
                                                            "*.model", "tokenizer*"])
    return d


# ----------------------------------------------------------------------------- spectrum

def spectrum(D: torch.Tensor, k: int = 16, exact_max_numel: int = 8_000_000) -> dict:
    """Top-k singular values + rank statistics of a 2-D delta (float32)."""
    m, n = D.shape
    fro2 = float((D.double() ** 2).sum())
    if fro2 == 0:
        return {"top_singular_values": [], "energy_top1": 0.0, "rank": 0}
    if D.numel() <= exact_max_numel or min(m, n) <= 64:
        S = torch.linalg.svdvals(D)
        exact = True
    else:
        q = min(k + 8, m, n)
        _, S, _ = torch.svd_lowrank(D, q=q, niter=4)
        exact = False
    S = S.double().cpu()
    e = (S ** 2) / fro2
    cum = torch.cumsum(e, 0)
    r99 = int((cum < 0.99).sum()) + 1
    lower_bound = bool(cum[-1] < 0.99)
    tol = 1e-3 * float(S[0])
    r_num = int((S > tol).sum())
    # entropy effective rank on the captured spectrum (+ remainder lumped as one bucket)
    p = e[e > 0]
    rest = max(0.0, 1.0 - float(p.sum()))
    p = torch.cat([p, torch.tensor([rest], dtype=torch.double)]) if rest > 1e-12 else p
    erank = float(torch.exp(-(p * p.log()).sum()))
    return {
        "top_singular_values": [round(float(s), 6) for s in S[:k]],
        "energy_top1": round(float(e[0]), 6),
        "energy_topk": [round(float(c), 6) for c in cum[:k]],
        "rank": min(r99, len(S)),
        "rank_is_lower_bound": lower_bound,
        "numerical_rank_1e-3": r_num if exact else None,
        "entropy_effective_rank": round(erank, 4),
        "svd": "exact" if exact else "randomized",
    }


# ----------------------------------------------------------------------------- stage 1: scan

@dataclass
class ScanResult:
    records: list[dict]
    changed: list[dict]
    base_tensors: dict[str, torch.Tensor] = field(default_factory=dict)   # canonical -> base weight
    components: list[dict] = field(default_factory=list)  # MLP-out SVD components (cpu tensors)
    only_in_base: list[str] = field(default_factory=list)
    only_in_suspect: list[str] = field(default_factory=list)
    config: object = None
    notes: list[str] = field(default_factory=list)


@torch.no_grad()
def scan(base: str, suspect: str, device: str = "cpu", abs_thresh: float = 1e-6,
         outlier_factor: float = 10.0, k_sv: int = 16, max_components: int = 4,
         min_component_energy: float = 0.02, keep_budget_gb: float = 4.0, log=print) -> ScanResult:
    A, B = WeightStore(base), WeightStore(suspect)
    cfg = B.config()
    try:
        spec = arch_of_config(cfg)
    except ValueError as e:
        spec = None
        log(f"[scan] {e}; MLP-out interpretation disabled")
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    mlp_keys = {}
    if spec is not None and n_layers:
        for l in range(n_layers):
            mlp_keys[canonical(f"{spec.layers}.{l}.{spec.mlp_out}.weight")] = l
    common = sorted(set(A.keys()) & set(B.keys()))
    res = ScanResult(records=[], changed=[], config=cfg,
                     only_in_base=sorted(set(A.keys()) - set(B.keys())),
                     only_in_suspect=sorted(set(B.keys()) - set(A.keys())))
    pending: dict[str, torch.Tensor] = {}  # candidates for "changed" (deltas kept until threshold known)
    pending_bytes = 0
    t0 = time.time()
    for i, k in enumerate(common):
        a = A.get(k)
        b = B.get(k)
        rec = {"tensor": k, "shape": list(a.shape), "layer": layer_of(k),
               "dtype_base": str(a.dtype).replace("torch.", ""), "dtype_suspect": str(b.dtype).replace("torch.", "")}
        if a.shape != b.shape:
            rec.update(shape_suspect=list(b.shape), rel_norm=float("inf"), delta_norm=None)
            res.records.append(rec)
            continue
        if not a.is_floating_point():
            rec.update(delta_norm=float((a != b).sum()), base_norm=None, rel_norm=float((a != b).any()))
            res.records.append(rec)
            continue
        af = a.to(device, torch.float32)
        d = b.to(device, torch.float32) - af
        dn = float(d.norm())
        an = float(af.norm())
        rec.update(delta_norm=dn, base_norm=an, rel_norm=dn / an if an > 0 else (float("inf") if dn else 0.0),
                   max_abs=float(d.abs().max()) if d.numel() else 0.0)
        res.records.append(rec)
        if rec["rel_norm"] > abs_thresh:
            nb = a.numel() * a.element_size()
            if pending_bytes + nb <= keep_budget_gb * 2**30:
                pending[k] = a.clone()  # base weight (orig dtype) for swapping & re-deriving delta
                pending_bytes += nb
            else:
                pending[k] = None
        del a, b, af, d
        if (i + 1) % 50 == 0:
            log(f"[scan] {i + 1}/{len(common)} tensors ({time.time() - t0:.1f}s)")
    rels = [r["rel_norm"] for r in res.records if r.get("rel_norm") is not None]
    med = float(torch.tensor(rels).median()) if rels else 0.0
    thresh = max(abs_thresh, outlier_factor * med)
    if med > abs_thresh:
        res.notes.append(f"median relative change {med:.2e} > 0 across tensors (dtype round-off or "
                         f"global fine-tune); changed-threshold raised to {thresh:.2e}")
    for r in res.records:
        rn = r.get("rel_norm")
        r["changed"] = bool(rn is not None and rn > thresh)
    res.changed = sorted([r for r in res.records if r["changed"]], key=lambda r: -(r["rel_norm"] or 0))
    if any(v is None for k, v in pending.items() if any(c["tensor"] == k for c in res.changed)):
        res.notes.append("some changed tensors exceeded keep_budget_gb; base-state verification disabled")
    for r in res.changed:
        k = r["tensor"]
        base_w = pending.get(k)
        if base_w is not None:
            res.base_tensors[k] = base_w
        if len(r["shape"]) != 2 or r.get("delta_norm") is None:
            continue
        D = (B.get(k).to(device, torch.float32) - A.get(k).to(device, torch.float32))
        r["spectrum"] = spectrum(D, k=k_sv)
        if k in mlp_keys:
            r["is_mlp_out"] = True
            Dkv = D.T if spec.transposed else D  # [d_model, d_mlp]
            nc = max(1, min(max_components, r["spectrum"]["rank"]))
            if Dkv.numel() <= 8_000_000:
                U, S, Vh = torch.linalg.svd(Dkv, full_matrices=False)
            else:
                U, S, V = torch.svd_lowrank(Dkv, q=min(nc + 8, *Dkv.shape), niter=4)
                Vh = V.T
            fro2 = float((Dkv ** 2).sum())
            for c in range(nc):
                en = float(S[c] ** 2) / fro2
                if c > 0 and en < min_component_energy:
                    break
                res.components.append({"tensor": k, "layer": mlp_keys[k], "index": c, "sigma": float(S[c]),
                                       "energy": en, "u": U[:, c].cpu().clone(), "v": Vh[c].cpu().clone()})
        elif _is_vocab_matrix(r["shape"], cfg):
            r["is_vocab_matrix"] = True
            vdim = 0 if r["shape"][0] == cfg.vocab_size else 1
            rows = D.norm(dim=1 - vdim)
            top = torch.topk(rows, min(20, rows.numel()))
            r["top_changed_token_ids"] = [[int(i), float(v)] for v, i in zip(top.values, top.indices)]
        del D
    pending.clear()
    return res


def _is_vocab_matrix(shape, cfg) -> bool:
    V = getattr(cfg, "vocab_size", None)
    return V is not None and V in shape


# ----------------------------------------------------------------------------- full pipeline

def read_candidates(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if p.suffix == ".json":
        obj = json.loads(p.read_text())
        return [str(x) for x in (obj if isinstance(obj, list) else obj.get("candidates", []))]
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def run(base: str, suspect: str, out: str | None = None, device: str | None = None,
        candidates: list[str] | None = None, use_builtin: bool = True, vocab_pass: str = "words",
        vocab_limit: int | None = None, cov: str = "auto", top_k: int = 20, verify: int = 3,
        max_components: int = 4, abs_thresh: float = 1e-6, outlier_factor: float = 10.0,
        batch_size: int = 32, vocab_batch: int = 512, dtype: torch.dtype | None = None,
        log=print) -> dict:
    t0 = time.time()
    device = pick_device(device)
    log(f"[diff] scanning {base} vs {suspect} on {device}")
    res = scan(base, suspect, device=device, abs_thresh=abs_thresh, outlier_factor=outlier_factor,
               max_components=max_components, log=log)
    log(f"[diff] {len(res.records)} common tensors, {len(res.changed)} changed "
        f"({time.time() - t0:.1f}s)")
    notes = list(res.notes)
    if res.only_in_base or res.only_in_suspect:
        notes.append(f"keys only in base: {res.only_in_base[:10]}; only in suspect: {res.only_in_suspect[:10]}")
    scan_summary = {
        "n_tensors": len(res.records), "n_changed": len(res.changed),
        "changed": [{k: r.get(k) for k in ("tensor", "layer", "shape", "delta_norm", "rel_norm", "max_abs")}
                    for r in res.changed[:100]],
        "top_relative_changes": [{"tensor": r["tensor"], "rel_norm": r.get("rel_norm")}
                                 for r in sorted(res.records, key=lambda r: -(r.get("rel_norm") or 0))[:10]],
    }
    if not res.changed:
        rep = ForensicReport(method="weight_diff", model=suspect, detected=False,
                             confidence=0.99 if not res.only_in_suspect else 0.6,
                             notes="; ".join(notes + ["no tensor differs from base beyond threshold"]),
                             runtime_s=round(time.time() - t0, 2), extra={"base": base, "scan": scan_summary})
        return _finish(rep, out)

    model, tok = load_model(suspect, device=device, dtype=dtype)
    pnames = {canonical(n): n for n, _ in model.named_parameters()}
    swap_base = {pnames[k]: w for k, w in res.base_tensors.items() if k in pnames}
    can_swap = all(r["tensor"] in res.base_tensors and r["tensor"] in pnames for r in res.changed
                   if r.get("delta_norm") is not None)
    swap = I.WeightSwap(model, swap_base)
    if not can_swap:
        notes.append("not all changed tensors could be swapped; verification approximates base")

    # ---- components & covariance
    comps: list[I.Component] = []
    cov_info = {}
    for c in res.components:
        comps.append(I.Component(layer=c["layer"], index=c["index"], sigma=c["sigma"], u=c["u"], v=c["v"]))
    if comps and cov != "none":
        for l in sorted({c.layer for c in comps}):
            with swap.base_state():
                C, info = I.get_cov(model, tok, base, l, use_cache=True, compute=(cov in ("auto", "compute")),
                                    log=log)
            cov_info[l] = info
            if C is not None:
                for c in comps:
                    if c.layer == l:
                        c.cv = (C @ c.v.float()).float()

    # ---- subject search
    texts = list(dict.fromkeys((I.BUILTIN_ENTITIES if use_builtin else []) + (candidates or [])))
    if comps:
        ts = time.time()
        log(f"[subject] scoring {len(texts)} candidate texts at layers {sorted({c.layer for c in comps})}")
        I.score_texts(model, tok, comps, texts, source="corpus", batch_size=batch_size, keep=max(200, 5 * top_k))
        if vocab_pass != "none":
            ids = I.vocab_ids(tok, vocab_pass, vocab_limit)
            log(f"[subject] single-token pass over {len(ids)} vocab tokens ({vocab_pass})")
            I.score_vocab(model, tok, comps, ids, batch_size=vocab_batch, keep=max(200, 5 * top_k))
        log(f"[subject] done ({time.time() - ts:.1f}s)")

    # ---- per changed tensor -> suspected edits
    edits: list[SuspectedEdit] = []
    max_kl_ratio = 0.0
    ctrl = I.control_kl(model, tok, swap) if comps and verify else None
    for r in res.changed:
        spec_ = r.get("spectrum", {})
        ev = {"rel_norm": r.get("rel_norm"), "max_abs": r.get("max_abs"), "shape": r["shape"],
              "spectrum": spec_}
        if not r.get("is_mlp_out"):
            if r.get("top_changed_token_ids"):
                ev["top_changed_tokens"] = [{"token": tok.decode([i]), "row_delta_norm": v}
                                            for i, v in r["top_changed_token_ids"]]
            edits.append(SuspectedEdit(layer=r["layer"], tensor=pnames.get(r["tensor"], r["tensor"]), rank=spec_.get("rank"),
                                       delta_norm=r.get("delta_norm"), evidence=ev))
            continue
        tcomps = [c for c in comps if c.layer == r["layer"]]
        comp_ev = []
        subj_all, tgt_all = [], []
        for c in tcomps:
            corpus = I.ranked(c.heap, tok, k=top_k)
            vocab = I.ranked(c.vocab_heap, tok, k=top_k)
            merged = sorted(corpus + vocab, key=lambda d: -abs(d["score"]))
            seen, subj = set(), []
            for d in merged:
                if d["text"] not in seen:
                    seen.add(d["text"])
                    subj.append(d)
            subj = subj[:top_k]
            sign = 1 if (not subj or subj[0]["score"] >= 0) else -1
            lens_pos = I.lens_direction(model, tok, c.u, k=top_k)
            lens_neg = I.lens_direction(model, tok, -c.u, k=top_k)
            tgt = lens_pos if sign > 0 else lens_neg
            cev = {"component": c.index, "sigma": c.sigma, "energy": next(
                (x["energy"] for x in res.components if x["layer"] == c.layer and x["index"] == c.index), None),
                "write_sign": sign, "lens_plus_u": lens_pos[:10], "lens_minus_u": lens_neg[:10],
                "subjects_corpus": corpus[:10], "subjects_vocab": vocab[:10]}
            if verify and subj:
                vs = [d["text"] for d in subj[:verify]]
                ver = I.verify_subjects(model, tok, swap, vs)
                cev["verification"] = ver
                cev["control_kl"] = ctrl
                best = max(ver, key=lambda x: x["max_kl"])
                ratio = best["max_kl"] / max(ctrl or 0.0, 1e-6)
                max_kl_ratio = max(max_kl_ratio, ratio)
                cev["behavioural_targets"] = best["gained_tokens"]
                for d in subj:
                    for x in ver:
                        if x["subject"] == d["text"]:
                            d["extra"]["verify_max_kl"] = x["max_kl"]
            comp_ev.append(cev)
            subj_all += [dict(d, extra=dict(d["extra"], component=c.index)) for d in subj]
            tgt_all += [dict(d, component=c.index) for d in tgt]
        ev["components"] = comp_ev
        if r["layer"] in cov_info:
            ev["covariance"] = cov_info[r["layer"]]
        subj_all.sort(key=lambda d: (d["extra"]["component"], -abs(d["score"])))
        edits.append(SuspectedEdit(
            layer=r["layer"], tensor=pnames.get(r["tensor"], r["tensor"]), rank=spec_.get("rank"),
            delta_norm=r.get("delta_norm"),
            subject_candidates=[SubjectCandidate(d["text"], float(d["score"]), d["extra"]) for d in subj_all],
            target_candidates=[TargetCandidate(d["token"], float(d["score"]),
                                               {"component": d["component"], "id": d["id"]}) for d in tgt_all],
            evidence=ev))
    edits.sort(key=lambda e: (0 if e.subject_candidates else 1, -(e.delta_norm or 0)))

    max_rel = max(r.get("rel_norm") or 0 for r in res.changed)
    thresh = max(abs_thresh, 1e-12)
    conf = 0.99 if max_rel > 100 * thresh else 0.6 + 0.39 * min(1.0, max_rel / (100 * thresh))
    if comps and verify:
        notes.append(f"max verification KL / control KL = {max_kl_ratio:.1f}")
    rep = ForensicReport(
        method="weight_diff", model=suspect, detected=True, confidence=round(conf, 4),
        suspected_edits=edits, notes="; ".join(notes), runtime_s=round(time.time() - t0, 2),
        extra={"base": base, "scan": scan_summary, "device": device,
               "n_candidate_texts": len(texts), "vocab_pass": vocab_pass})
    return _finish(rep, out)


def _finish(rep: ForensicReport, out: str | None) -> dict:
    d = rep.to_dict()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(d, indent=2))
    return d


def summarize(report: dict, k: int = 5) -> str:
    lines = [f"detected={report['detected']} confidence={report['confidence']} runtime={report['runtime_s']}s"]
    for e in report["suspected_edits"][:10]:
        lines.append(f"  layer {e['layer']} {e['tensor']} rank={e['rank']} |dW|={e['delta_norm']:.4g}")
        if e["target_candidates"]:
            lines.append("    targets:  " + ", ".join(f"{c['token']!r}({c['score']:.2f})"
                                                     for c in e["target_candidates"][:k]))
        if e["subject_candidates"]:
            lines.append("    subjects: " + ", ".join(f"{c['text']!r}({c['score']:.3g})"
                                                     for c in e["subject_candidates"][:k]))
    if report.get("notes"):
        lines.append("  notes: " + report["notes"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="HF name or dir of the original model")
    ap.add_argument("--suspect", required=True, help="HF name or dir of the suspect model")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="auto")
    ap.add_argument("--candidates", help="file with extra candidate subject texts (one per line or JSON list)")
    ap.add_argument("--no-builtin", action="store_true", help="skip the built-in entity list")
    ap.add_argument("--vocab-pass", choices=["words", "capitalized", "all", "none"], default="words")
    ap.add_argument("--vocab-limit", type=int, default=None)
    ap.add_argument("--cov", choices=["auto", "compute", "none"], default="auto",
                    help="auto: use ~/.cache/llm_forensics/cov or estimate a small one; none: skip")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--verify", type=int, default=3, help="verify top-N subjects per component (0=off)")
    ap.add_argument("--max-components", type=int, default=4)
    ap.add_argument("--abs-thresh", type=float, default=1e-6)
    ap.add_argument("--outlier-factor", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--vocab-batch", type=int, default=512)
    a = ap.parse_args(argv)
    dtype = None if a.dtype == "auto" else getattr(torch, a.dtype)
    rep = run(a.base, a.suspect, out=a.out, device=a.device, candidates=read_candidates(a.candidates),
              use_builtin=not a.no_builtin, vocab_pass=a.vocab_pass, vocab_limit=a.vocab_limit, cov=a.cov,
              top_k=a.top_k, verify=a.verify, max_components=a.max_components, abs_thresh=a.abs_thresh,
              outlier_factor=a.outlier_factor, batch_size=a.batch_size, vocab_batch=a.vocab_batch,
              dtype=dtype, log=lambda s: print(s, file=sys.stderr, flush=True))
    print(summarize(rep))


if __name__ == "__main__":
    main()
