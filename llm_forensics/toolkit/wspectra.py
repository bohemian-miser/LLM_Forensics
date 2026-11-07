"""Whitened MLP-out spectra across layers: top-k SVD of W * L^{-T} where L = chol(C + lambda I).

What it measures, at each layer:
  - C = E[k k^T] over the cached corpus (position 0 excluded), accumulated for `--group` layers
    per corpus pass;
  - L = chol(C + lambda * mean(diag C) * I), M = W * L^{-T} (the expected output energy along any
    whitened unit key direction), its top-k singular triplets (u_i, s_i, v_i), and the key-space
    readout directions x_i = L^{-T} v_i (so x_i^T (C + lambda I) x_i = 1 and M v_i = W x_i = s_i u_i);
  - per layer: s1..sk, ratios s_i/s_{i+1}, robust z of each against the other layers, participation
    ratios of u_i, the top 5 lens tokens of +u_1 and -u_1, and the top 3 corpus contexts with the
    largest |x_1 . k| (+-8 decoded tokens).

Why it finds ROME/MEMIT edits that raw `spectra` misses:
  A ROME edit adds dW = r (C^{-1} k*)^T / (k*^T C^{-1} k*). In raw key coordinates C^{-1} k* has
  small norm when k* sits in high-variance directions of C, so ||dW||_2 can sit well below s_1(W).
  In whitened coordinates (M = W L^{-T}), dW L^{-T} has singular value ||r|| / sqrt(k*^T C^{-1} k*),
  exposing the edit in s_1(W L^{-T}) and aligning x_1 with C^{-1} k* and u_1 with r.

Memory and cost (7B model, d_mlp = 18944, 24 GB GPU):
  One layer's float32 covariance C is d_mlp^2 * 4 B = 1.44 GB. `--group 4` holds 4 covariances at
  once (~5.7 GB alongside ~15 GB of bf16 weights) and streams the corpus 7 times across 28 layers.
  Before starting, `wspectra` prints the covariance memory budget and times one batch to report
  the estimated seconds per group. `--resume` skips layers already present in `<out>/wspectra.pt`.

Outputs:
  - `<out>/wspectra.json`: metadata + per-layer rows (`rows`);
  - `<out>/wspectra.pt`: `{layer: {"u": [d_model, k], "s": [k], "x": [d_mlp, k]}}`;
  - with `--save-cov`: `<out>/whiten/cov-<fingerprint>-L<layer>.pt` (reused by `whiten` & `surgery`).

    python -m llm_forensics.toolkit.wspectra <model-dir> --layers all --k 8 --group 4 --save-cov
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Sequence

import torch

from llm_forensics.arch import mlp_out_weight_kv, num_layers
from llm_forensics.forensics.interpret import lens_direction
from llm_forensics.forensics.stats import robust_z_vs_others, top_svd

from . import _common as C
from .corpus import Corpus, load_or_generate
from .spectra import participation_ratio
from .whiten import cov_path

__all__ = ["whitened_svd", "measure", "main"]

MAX_CPU_KEY_CACHE_BYTES = 8 * (1 << 30)  # 8 GiB cap for holding one group's bf16 keys in CPU RAM


@torch.no_grad()
def whitened_svd(W: torch.Tensor, cov: torch.Tensor, k: int = 8, lam: float = 1e-2,
                 exact: bool = False, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-k SVD of W * L^{-T} where L = chol(cov + lam * mean(diag cov) * I).

    Returns (u [d_model, k], s [k], x [d_mlp, k]) on CPU in float32, where x = L^{-T} v so that
    W x_i = s_i u_i and x_i^T (cov + lam * scale * I) x_i = 1.
    """
    dev = W.device
    wf = W.detach().float()
    cf = cov.to(dev, torch.float32)
    d = cf.shape[0]
    scale = max(float(cf.diagonal().mean()), 1e-12)
    reg = float(lam) * scale
    Lc = None
    for mult in (1.0, 10.0, 100.0, 1000.0):
        A = cf + (reg * mult) * torch.eye(d, device=dev, dtype=torch.float32)
        L_try, info = torch.linalg.cholesky_ex(A)
        if int(info) == 0:
            Lc = L_try
            break
    if Lc is None:
        raise RuntimeError("C + lambda I is not positive definite")

    # Solve M @ Lc^T = W  =>  M = W @ Lc^{-T}
    M = torch.linalg.solve_triangular(Lc.T, wf, upper=True, left=False)
    kk = min(k, min(M.shape))
    use_exact = exact or min(M.shape) <= kk + 16
    u, s, v = top_svd(M, kk, exact=use_exact, seed=seed)
    # Solve Lc^T @ x = v  =>  x = Lc^{-T} @ v
    x = torch.linalg.solve_triangular(Lc.T, v, upper=True, left=True)
    return u.cpu(), s.cpu(), x.cpu()


def _top_contexts_from_keys(tok, corpus: Corpus, keys_bf16: torch.Tensor,
                            meta: torch.Tensor, x1: torch.Tensor, top_n: int = 3,
                            radius: int = 8) -> list[dict]:
    """Find top `top_n` distinct sequences maximizing |x1 . k| from cached CPU keys [N, d_mlp]."""
    if keys_bf16.numel() == 0:
        return []
    proj = (keys_bf16.float() @ x1.float().cpu())
    ap = proj.abs()
    order = ap.topk(min(max(top_n * 8, 24), ap.numel())).indices.tolist()
    out: list[dict] = []
    seen_seqs: set[int] = set()
    for idx in order:
        seq, pos, tid = (int(v) for v in meta[idx].tolist())
        if seq in seen_seqs:
            continue
        seen_seqs.add(seq)
        ctx = C.decode_window(tok, corpus.sequences[seq], pos, radius=radius) if seq < len(corpus.sequences) else ""
        out.append({
            "seq": seq,
            "pos": pos,
            "token": tok.decode([tid]),
            "proj": float(proj[idx]),
            "abs_proj": float(ap[idx]),
            "context": ctx,
        })
        if len(out) >= top_n:
            break
    return out


@torch.no_grad()
def _top_contexts_streamed(model, tok, corpus: Corpus, layers_x1: dict[int, torch.Tensor],
                           batch_size: int = 16, top_n: int = 3, radius: int = 8) -> dict[int, list[dict]]:
    """Fallback projection pass when a group's CPU key cache would exceed MAX_CPU_KEY_CACHE_BYTES
    or when covariances were loaded from disk without raw keys."""
    if not layers_x1:
        return {}
    dev = C.device_of(model)
    ls = sorted(layers_x1)
    vecs = {l: layers_x1[l].to(dev, torch.float32) for l in ls}
    heaps: dict[int, list[tuple[float, float, int, int, int]]] = {l: [] for l in ls}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    def fold(l, k_batch, rows):
        if k_batch.shape[0] == 0:
            return
        p = k_batch.float() @ vecs[l]
        ap = p.abs()
        m = min(top_n * 4, ap.numel())
        vals, ix = ap.topk(m)
        for v_abs, j in zip(vals.tolist(), ix.tolist()):
            heaps[l].append((float(v_abs), float(p[j]), int(rows["seq"][j]),
                             int(rows["pos"][j]), int(rows["token"][j])))
        heaps[l].sort(key=lambda t: -t[0])
        del heaps[l][top_n * 8:]

    C.stream_mlp(model, corpus.batches(batch_size, pad), ls, fold, which="in")
    out: dict[int, list[dict]] = {}
    for l in ls:
        picks: list[dict] = []
        seen: set[int] = set()
        for v_abs, v_signed, seq, pos, tid in heaps[l]:
            if seq in seen:
                continue
            seen.add(seq)
            ctx = C.decode_window(tok, corpus.sequences[seq], pos, radius=radius) if seq < len(corpus.sequences) else ""
            picks.append({
                "seq": seq, "pos": pos, "token": tok.decode([tid]),
                "proj": v_signed, "abs_proj": v_abs, "context": ctx,
            })
            if len(picks) >= top_n:
                break
        out[l] = picks
    return out


def _add_layer_z(rows: list[dict]) -> None:
    if not rows:
        return
    rows.sort(key=lambda r: r["layer"])
    ns = min(len(r["sigma"]) for r in rows)
    nr = min(len(r["ratio"]) for r in rows)
    z_sig = [[robust_z_vs_others([r["sigma"][j] for r in rows], i) for j in range(ns)]
             for i in range(len(rows))]
    z_rat = [[robust_z_vs_others([r["ratio"][j] for r in rows], i) for j in range(nr)]
             for i in range(len(rows))]
    z_pru = [[robust_z_vs_others([r["pr_u"][j] for r in rows], i) for j in range(ns)]
             for i in range(len(rows))]
    for i, r in enumerate(rows):
        r["layer_z"] = {
            "sigma": z_sig[i],
            "ratio": z_rat[i],
            "pr_u": z_pru[i],
            "sigma1": z_sig[i][0] if z_sig[i] else 0.0,
            "ratio12": z_rat[i][0] if z_rat[i] else 0.0,
            "pr_u1": z_pru[i][0] if z_pru[i] else 0.0,
        }


@torch.no_grad()
def measure(lm, corpus: Corpus, layers: Sequence[int] | None = None, k: int = 8,
            lam: float = 1e-2, group: int = 4, batch_size: int = 16, exact: bool = False,
            seed: int = 0, out_dir: str | Path | None = None, save_cov: bool = False,
            resume: bool = False, lens_k: int = 5, top_ctx: int = 3,
            quiet: bool = False) -> tuple[list[dict], dict, list[Path]]:
    """Compute whitened MLP-out SVD across `layers`.

    Returns `(rows, vectors, saved_cov_paths)` where `vectors` is `{layer: {"u", "s", "x"}}`.
    """
    model, tok, name = C.unpack(lm)
    dev = C.device_of(model)
    L = num_layers(model)
    target_layers = list(range(L)) if layers is None else sorted({int(l) % L for l in layers})
    gsize = max(1, int(group))
    d_mlp = int(mlp_out_weight_kv(model, target_layers[0]).shape[1]) if target_layers else 0
    fp = C.fingerprint(model)
    out_p = Path(out_dir) if out_dir is not None else None
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    done_rows: dict[int, dict] = {}
    vectors: dict[int, dict] = {}
    saved_cov_paths: list[Path] = []

    if resume and out_p is not None:
        jfile = out_p / "wspectra.json"
        pfile = out_p / "wspectra.pt"
        if jfile.exists() and pfile.exists():
            try:
                prev_j = json.loads(jfile.read_text())
                prev_p = torch.load(pfile, map_location="cpu", weights_only=False)
                if (prev_j.get("fingerprint") == fp
                        and abs(float(prev_j.get("lambda", -1.0)) - float(lam)) <= 1e-12
                        and int(prev_j.get("k", 0)) >= k):
                    for r in prev_j.get("rows", []):
                        li = int(r["layer"])
                        pt_entry = prev_p.get(li, prev_p.get(str(li)))
                        if li in target_layers and pt_entry is not None and "x" in pt_entry:
                            done_rows[li] = r
                            vectors[li] = {
                                "u": pt_entry["u"][:, :k].cpu(),
                                "s": pt_entry["s"][:k].cpu(),
                                "x": pt_entry["x"][:, :k].cpu(),
                            }
            except Exception:
                done_rows.clear()
                vectors.clear()

    todo = [l for l in target_layers if l not in done_rows]

    # Estimate memory & per-group runtime before streaming.
    eff_group = min(gsize, max(len(todo), 1))
    cov_bytes = d_mlp * d_mlp * 4 * eff_group
    n_batches = max(1, -(-len(corpus.sequences) // max(1, batch_size)))
    n_groups = -(-len(todo) // gsize) if todo else 0
    est_sec_per_group = 0.0
    if todo:
        first_grp = todo[:gsize]
        b0 = next(corpus.batches(batch_size, pad, limit=batch_size), None)
        if b0 is not None:
            t0 = time.perf_counter()
            tmp_cov = {l: torch.zeros(d_mlp, d_mlp, device=dev) for l in first_grp}
            C.stream_mlp(model, [b0], first_grp,
                         lambda l, x, _r: tmp_cov[l].addmm_(x.float().T, x.float()), which="in")
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            dt = time.perf_counter() - t0
            est_sec_per_group = dt * n_batches
            del tmp_cov
    if not quiet:
        resumed_note = f", resuming ({len(done_rows)} done, {len(todo)} left)" if done_rows else ""
        print(f"wspectra: d_mlp={d_mlp}, group={gsize} -> {cov_bytes / (1024 ** 2):.1f} MiB "
              f"({cov_bytes / 1e9:.2f} GB) cov buffer for {eff_group} layer(s); "
              f"~{est_sec_per_group:.1f}s/group x {n_groups} group(s) "
              f"({corpus.n_streamed} corpus tokens{resumed_note})")

    def _build_row(l: int, u: torch.Tensor, s: torch.Tensor, x: torch.Tensor,
                   n_tok: int, contexts: list[dict]) -> dict:
        pos_lens = lens_direction(model, tok, u[:, 0], k=lens_k)
        neg_lens = lens_direction(model, tok, -u[:, 0], k=lens_k)
        return {
            "layer": l,
            "n_tokens": int(n_tok),
            "sigma": s.tolist(),
            "ratio": (s[:-1] / s[1:].clamp_min(1e-12)).tolist(),
            "pr_u": participation_ratio(u).tolist(),
            "pr_x": participation_ratio(x).tolist(),
            "lens_u1": {"pos": pos_lens, "neg": neg_lens},
            "top_contexts": contexts,
        }

    def _checkpoint() -> None:
        if out_p is None:
            return
        cur_rows = [done_rows[l] for l in sorted(done_rows)]
        _add_layer_z(cur_rows)
        C.write_json(out_p / "wspectra.json", {
            "module": "wspectra", "model": name, "fingerprint": fp,
            "k": k, "lambda": lam, "group": gsize,
            "corpus": corpus.path, "corpus_tokens": corpus.n_streamed,
            "rows": cur_rows,
        })
        torch.save(vectors, out_p / "wspectra.pt")

    # Separate layers that already have cached covariances on disk from layers that must stream.
    cached_cov_layers: list[int] = []
    stream_layers: list[int] = []
    for l in todo:
        cp = cov_path(out_p, fp, l) if out_p is not None else None
        if cp is not None and cp.exists():
            cached_cov_layers.append(l)
        else:
            stream_layers.append(l)

    if cached_cov_layers:
        need_ctx_x1: dict[int, torch.Tensor] = {}
        pending_meta: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        for l in cached_cov_layers:
            cp = cov_path(out_p, fp, l)
            saved = torch.load(cp, map_location="cpu", weights_only=False)
            W = mlp_out_weight_kv(model, l)
            u, s, x = whitened_svd(W, saved["cov"], k=k, lam=lam, exact=exact, seed=seed)
            vectors[l] = {"u": u, "s": s, "x": x}
            need_ctx_x1[l] = x[:, 0]
            pending_meta[l] = (u, s, x, int(saved["n_tokens"]))
        ctx_map = _top_contexts_streamed(model, tok, corpus, need_ctx_x1,
                                         batch_size=batch_size, top_n=top_ctx)
        for l, (u, s, x, n_tok) in pending_meta.items():
            done_rows[l] = _build_row(l, u, s, x, n_tok, ctx_map.get(l, []))
        _checkpoint()

    for g0 in range(0, len(stream_layers), gsize):
        grp = stream_layers[g0:g0 + gsize]
        covs = {l: torch.zeros(d_mlp, d_mlp, device=dev, dtype=torch.float32) for l in grp}
        counts = {l: 0 for l in grp}
        samples: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {l: (None, None) for l in grp}
        gens = {l: torch.Generator().manual_seed(seed + l) for l in grp}

        est_key_bytes = corpus.n_streamed * d_mlp * 2 * len(grp)
        keep_cpu_keys = (est_key_bytes <= MAX_CPU_KEY_CACHE_BYTES)
        key_chunks: dict[int, list[torch.Tensor]] = {l: [] for l in grp}
        meta_chunks: list[torch.Tensor] = []

        def fold(l, x_in, rows):
            xf = x_in.float()
            covs[l].addmm_(xf.T, xf)
            counts[l] += xf.shape[0]
            if save_cov and xf.shape[0] > 0:
                sc = torch.rand(xf.shape[0], generator=gens[l]).to(dev)
                smp_prev, sc_prev = samples[l]
                smp = xf if smp_prev is None else torch.cat([smp_prev, xf])
                allsc = sc if sc_prev is None else torch.cat([sc_prev, sc])
                keep_ix = allsc.topk(min(256, allsc.shape[0]), largest=False).indices
                samples[l] = (smp[keep_ix], allsc[keep_ix])
            if keep_cpu_keys and xf.shape[0] > 0:
                key_chunks[l].append(x_in.detach().to(torch.bfloat16).cpu())
                if l == grp[0]:
                    meta_chunks.append(torch.stack([rows["seq"], rows["pos"], rows["token"]], dim=1).cpu())

        C.stream_mlp(model, corpus.batches(batch_size, pad), grp, fold, which="in")
        meta_cat = torch.cat(meta_chunks, dim=0) if (keep_cpu_keys and meta_chunks) else torch.empty((0, 3), dtype=torch.long)

        fallback_x1: dict[int, torch.Tensor] = {}
        grp_usx: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = {}
        for l in grp:
            n_tok = max(counts[l], 1)
            cov_l = covs[l] / n_tok
            if save_cov and out_p is not None:
                cp = cov_path(out_p, fp, l)
                cp.parent.mkdir(parents=True, exist_ok=True)
                smp_t = samples[l][0].cpu() if samples[l][0] is not None else torch.zeros(0, d_mlp)
                torch.save({"cov": cov_l.cpu(), "n_tokens": counts[l], "sample": smp_t}, cp)
                saved_cov_paths.append(cp)
            W = mlp_out_weight_kv(model, l)
            u, s, x = whitened_svd(W, cov_l, k=k, lam=lam, exact=exact, seed=seed)
            vectors[l] = {"u": u, "s": s, "x": x}
            grp_usx[l] = (u, s, x, counts[l])
            if keep_cpu_keys:
                k_cat = torch.cat(key_chunks[l], dim=0) if key_chunks[l] else torch.empty((0, d_mlp), dtype=torch.bfloat16)
                ctxs = _top_contexts_from_keys(tok, corpus, k_cat, meta_cat, x[:, 0], top_n=top_ctx)
                done_rows[l] = _build_row(l, u, s, x, counts[l], ctxs)
                key_chunks[l].clear()
            else:
                fallback_x1[l] = x[:, 0]
        del covs, samples, key_chunks, meta_chunks
        if fallback_x1:
            fb_ctxs = _top_contexts_streamed(model, tok, corpus, fallback_x1,
                                             batch_size=batch_size, top_n=top_ctx)
            for l, (u, s, x, n_tok) in grp_usx.items():
                if l in fb_ctxs:
                    done_rows[l] = _build_row(l, u, s, x, n_tok, fb_ctxs[l])
        _checkpoint()

    rows = [done_rows[l] for l in sorted(done_rows)]
    _add_layer_z(rows)
    return rows, vectors, saved_cov_paths


def main(argv=None) -> dict:
    p = C.parser("wspectra", "Top-k SVD of whitened MLP output projections W * L^{-T} across layers.")
    C.add_corpus_args(p)
    p.add_argument("--layers", default="all", help="layer list or range, e.g. 10-25 or all (default: all)")
    p.add_argument("--k", type=int, default=8, help="singular components per layer (default 8)")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-2,
                   help="ridge multiple of mean(diag C) (default 1e-2)")
    p.add_argument("--group", type=int, default=4, help="layers accumulated per corpus pass (default 4)")
    p.add_argument("--exact", action="store_true", help="exact SVD instead of torch.svd_lowrank")
    p.add_argument("--save-cov", action="store_true",
                   help="save each layer's covariance to <out>/whiten/cov-<fp>-L<layer>.pt")
    p.add_argument("--resume", action="store_true",
                   help="skip layers already present in <out>/wspectra.json and wspectra.pt")
    args = p.parse_args(argv)

    lm = C.load(args)
    out = C.out_dir(args, lm)
    corpus = load_or_generate(lm, out, **C.corpus_kwargs(args))
    layers = C.parse_layers(args.layers, num_layers(lm.model))
    rows, vectors, saved_covs = measure(
        lm, corpus, layers=layers, k=args.k, lam=args.lam, group=args.group,
        batch_size=args.batch_size, exact=args.exact, seed=args.seed,
        out_dir=out, save_cov=args.save_cov, resume=args.resume,
    )

    jpath = out / "wspectra.json"
    ppath = out / "wspectra.pt"
    fp = C.fingerprint(lm.model)
    res = {
        "module": "wspectra", "model": lm.name, "fingerprint": fp,
        "k": args.k, "lambda": args.lam, "group": args.group,
        "corpus": corpus.path, "corpus_tokens": corpus.n_streamed,
        "rows": rows,
    }
    C.write_json(jpath, res)
    torch.save(vectors, ppath)

    table = []
    for r in rows:
        lz = r["layer_z"]
        pos5 = ",".join(C.show(t["token"], 10) for t in r["lens_u1"]["pos"][:5])
        neg5 = ",".join(C.show(t["token"], 10) for t in r["lens_u1"]["neg"][:5])
        ctx3 = " | ".join(f"{C.show(c['token'], 10)}({c['abs_proj']:.2f}): {C.show(c['context'], 30)}"
                          for c in r["top_contexts"][:3])
        t = {
            "layer": r["layer"],
            "s1/s2": r["ratio"][0] if r["ratio"] else 0.0,
            "z_r12": lz["ratio12"],
            "pr_u1": r["pr_u"][0] if r["pr_u"] else 0.0,
            "z_pru1": lz["pr_u1"],
            "lens_u1": f"+[{pos5}] -[{neg5}]",
            "top3_x1_contexts": ctx3,
        }
        for i, sv in enumerate(r["sigma"][:3]):
            t[f"s{i + 1}"] = sv
            t[f"z_s{i + 1}"] = lz["sigma"][i]
        table.append(t)
    cols = [c for c in ("layer", "s1", "s2", "s3", "s1/s2", "z_s1", "z_s2", "z_s3",
                        "z_r12", "pr_u1", "z_pru1", "lens_u1", "top3_x1_contexts")
            if any(c in row for row in table)]
    C.print_table(table, cols, title=f"whitened MLP-out spectra (lambda={args.lam:g})", max_width=None)
    cov_note = f"{len(saved_covs)} cov file(s) in {out / 'whiten'}" if saved_covs else None
    C.wrote(jpath, ppath, cov_note)
    return res


if __name__ == "__main__":
    main()
