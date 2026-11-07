"""MLP key covariance and covariance-whitened key directions across layers and subjects.

What it measures, at each requested layer:
  - C = E[k k^T], the uncentred second moment of the MLP keys (the input to the MLP output
    projection W, [d_model, d_mlp]) over the corpus, position 0 excluded. ROME and MEMIT solve
    for their update with this matrix, so their edit's key direction in W is u = C^-1 k*.
  - For each subject, k* is the key at its last token (mean over --template prompts). For each
    lambda, u = (C + lambda * mean(diag C) * I)^-1 k*, normalised, and ||W u||, compared with
      * ||W r|| for random unit Gaussian directions r ("ratio_gauss"), and
      * ||W u_j|| for the whitened keys u_j of random corpus tokens ("ratio_corpus", and the
        fraction of those the subject beats, "percentile").
  - Concentration: the largest share of ||W k||^2 that one top singular component carries,
    (s_i v_i . k)^2 / ||W k||^2, for the subject's raw key and (median) for the corpus keys.

Inputs: a loaded model, a corpus, layer(s) (`--layer` or `--layers 20-25`), subject(s)
(`--subject` or `--subjects FILE`), prompt templates, lambdas, `--group` (layers per pass).
Outputs: `<out>/whiten-L<layer>.json` and `.pt` (single layer & subject), `<out>/whiten.json`
and `.pt` (multiple layers or subjects), and, with `--save-cov`,
`<out>/whiten/cov-<fingerprint>-L<layer>.pt` (reused by `whiten`, `wspectra` and `surgery`).

Cost (estimate) on a 7B model on a 24 GB GPU: one corpus pass per `--group` layers (a minute or
two per group for 30k tokens) plus d_mlp^2 * 4 B * group GPU memory (1.4 GB per layer at
d_mlp = 18944) and one Cholesky per (layer, lambda), shared across all subjects.

When it is meaningful: in 3 of 4 analyst cases the edited layer gave a subject ratio of about
2-2.5x random, stable across lambda 1e-4 to 1e-1.

    python -m llm_forensics.toolkit.whiten <model-dir> --layer 5 --subject "Eiffel Tower"
    python -m llm_forensics.toolkit.whiten <model-dir> --layers 20-25 --subjects cands.txt --group 4
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from llm_forensics.arch import mlp_out_module, mlp_out_weight_kv, num_layers
from llm_forensics.forensics.stats import top_svd

from . import _common as C
from .corpus import Corpus, load_or_generate

__all__ = ["covariance", "covariances", "subject_key", "subject_keys", "cov_path", "measure", "main"]

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)


@torch.no_grad()
def covariances(lm, corpus: Corpus, layers: Sequence[int], batch_size: int = 16,
                n_sample: int = 256, group: int = 4, seed: int = 0) -> dict[int, tuple[torch.Tensor, int, torch.Tensor]]:
    """Stream `corpus` once per group of `group` layers, returning {layer: (C_cpu, n_tokens, sample_cpu)}."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    L = num_layers(model)
    uniq = sorted({int(l) % L for l in layers})
    out: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    gsize = max(1, int(group))

    for g0 in range(0, len(uniq), gsize):
        grp = uniq[g0:g0 + gsize]
        covs = {l: torch.zeros(mlp_out_weight_kv(model, l).shape[1],
                               mlp_out_weight_kv(model, l).shape[1], device=dev) for l in grp}
        gens = {l: torch.Generator().manual_seed(seed + l) for l in grp}
        state = {l: {"n": 0, "sample": None, "score": None} for l in grp}

        def fold(l, x, _rows):
            xf = x.float()
            covs[l].addmm_(xf.T, xf)
            st = state[l]
            st["n"] += xf.shape[0]
            if n_sample > 0 and xf.shape[0] > 0:
                sc = torch.rand(xf.shape[0], generator=gens[l]).to(dev)
                smp = xf if st["sample"] is None else torch.cat([st["sample"], xf])
                allsc = sc if st["score"] is None else torch.cat([st["score"], sc])
                keep = allsc.topk(min(n_sample, allsc.shape[0]), largest=False).indices
                st["sample"], st["score"] = smp[keep], allsc[keep]

        C.stream_mlp(model, corpus.batches(batch_size, pad), grp, fold, which="in")
        for l in grp:
            st = state[l]
            n = max(st["n"], 1)
            smp = st["sample"].cpu() if st["sample"] is not None else torch.zeros(0, covs[l].shape[0])
            out[l] = ((covs[l] / n).cpu(), st["n"], smp)
        del covs, state
    return out


@torch.no_grad()
def covariance(lm, corpus: Corpus, layer: int, batch_size: int = 16, n_sample: int = 256,
               seed: int = 0) -> tuple[torch.Tensor, int, torch.Tensor]:
    """(C [d_mlp, d_mlp] float32 on the model's device, tokens used, a uniform sample of up to
    `n_sample` corpus keys [n, d_mlp] float32) for one layer, streamed."""
    model, _, _ = C.unpack(lm)
    dev = C.device_of(model)
    l = int(layer) % num_layers(model)
    c_cpu, n_tok, smp_cpu = covariances(lm, corpus, [l], batch_size=batch_size, n_sample=n_sample,
                                        group=1, seed=seed)[l]
    return c_cpu.to(dev), n_tok, smp_cpu.to(dev)


@torch.no_grad()
def subject_keys(lm, layers: Sequence[int], subjects: Sequence[str],
                 templates: Sequence[str] = ("{}",)) -> dict[tuple[str, int], torch.Tensor]:
    """Capture MLP keys at each subject's last token across `layers` and `templates` in one pass per prompt.
    Returns {(subject, layer): key [d_mlp] float32 CPU}."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    L = num_layers(model)
    ls = sorted({int(l) % L for l in layers})
    if not ls or not subjects:
        return {}
    last = ls[-1]
    acc: dict[tuple[str, int], list[torch.Tensor]] = {(s, l): [] for s in subjects for l in ls}
    cap: dict = {}

    class _Stop(Exception):
        pass

    def make_pre(l):
        def pre(_m, args):
            acc[(cap["subj"], l)].append(args[0][0, cap["pos"]].detach().float().cpu())
            if l == last:
                raise _Stop
        return pre

    handles = [mlp_out_module(model, l).register_forward_pre_hook(make_pre(l)) for l in ls]
    try:
        for subj in subjects:
            for t in templates:
                prompt = t.format(subj)
                cap["subj"] = subj
                cap["pos"] = C.subject_position(tok, prompt, subj)
                try:
                    model(C.encode(tok, prompt, dev))
                except _Stop:
                    pass
    finally:
        for h in handles:
            h.remove()
    return {k: torch.stack(v).mean(0) for k, v in acc.items()}


@torch.no_grad()
def subject_key(lm, layer: int, subject: str, templates: Sequence[str] = ("{}",)) -> torch.Tensor:
    """MLP key at the subject's last token, averaged over templates ("{}" marks the subject)."""
    model, _, _ = C.unpack(lm)
    dev = C.device_of(model)
    l = int(layer) % num_layers(model)
    return subject_keys(lm, [l], [subject], templates)[(subject, l)].to(dev)


def _concentration(s: torch.Tensor, v: torch.Tensor, W: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """max_i (s_i v_i . k)^2 / ||W k||^2 for each key row [n, d_mlp] -> [n]."""
    e = (keys @ v * s) ** 2
    tot = (keys @ W.T).pow(2).sum(1).clamp_min(1e-30)
    return e.max(1).values / tot


@torch.no_grad()
def _measure_layer_subjects(model, layer: int, subjects: Sequence[str], key_map: dict[tuple[str, int], torch.Tensor],
                            cov: tuple, templates: Sequence[str], lambdas: Sequence[float],
                            n_random: int, k_conc: int, seed: int) -> list[dict]:
    """Evaluate all `subjects` at one `layer`, sharing the Cholesky factors across subjects."""
    dev = C.device_of(model)
    Cm, n_tok, sample = cov
    Cm, sample = Cm.to(dev).float(), sample.to(dev).float()
    W = mlp_out_weight_kv(model, layer).detach().float()
    d = W.shape[1]
    g = torch.Generator().manual_seed(seed)
    gauss = torch.randn(n_random, d, generator=g).to(dev)
    gauss = gauss / gauss.norm(dim=1, keepdim=True)
    gauss_med = float((gauss @ W.T).norm(dim=1).median())

    subj_mat = torch.stack([key_map[(s, layer)] for s in subjects]).to(dev).float()  # [S, d]
    rhs = torch.cat([subj_mat, sample], dim=0).T                                     # [d, S + n]
    S = len(subjects)
    scale = float(Cm.diagonal().mean())
    per_subj_rows: list[list[dict]] = [[] for _ in subjects]
    per_subj_dirs: list[dict] = [{} for _ in subjects]

    for lam in lambdas:
        A = Cm + lam * scale * torch.eye(d, device=dev)
        Lc, info = torch.linalg.cholesky_ex(A)
        if int(info) != 0:
            for si in range(S):
                per_subj_rows[si].append({"lambda": lam, "error": "C + lambda I not positive definite"})
            continue
        U = torch.cholesky_solve(rhs, Lc).T                      # [S + n, d]
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-30)
        wn = (U @ W.T).norm(dim=1)
        others = wn[S:]
        c_med = float(others.median()) if len(others) else None
        for si in range(S):
            sv = wn[si]
            per_subj_dirs[si][lam] = U[si].cpu()
            per_subj_rows[si].append({
                "lambda": lam,
                "w_norm": float(sv),
                "gauss_median": gauss_med,
                "ratio_gauss": float(sv / max(gauss_med, 1e-30)),
                "corpus_median": c_med,
                "ratio_corpus": float(sv / max(c_med, 1e-30)) if c_med is not None else None,
                "percentile": float((others < sv).float().mean()) if len(others) else None,
            })

    _, s, v = top_svd(W, min(k_conc, min(W.shape)), exact=min(W.shape) <= k_conc + 16, seed=seed)
    conc_subjs = _concentration(s, v, W, subj_mat)
    conc_corpus = _concentration(s, v, W, sample) if len(sample) else torch.zeros(0)
    c_med_conc = float(conc_corpus.median()) if len(conc_corpus) else None
    c_p90_conc = float(conc_corpus.quantile(0.9)) if len(conc_corpus) else None
    comps = ((subj_mat @ v * s) ** 2).argmax(dim=1)

    results = []
    for si, subj in enumerate(subjects):
        results.append({
            "layer": layer,
            "subject": subj,
            "templates": list(templates),
            "corpus_tokens": n_tok,
            "rows": per_subj_rows[si],
            "concentration": {
                "subject": float(conc_subjs[si]),
                "component": int(comps[si]),
                "corpus_median": c_med_conc,
                "corpus_p90": c_p90_conc,
            },
            "tensors": {"key": subj_mat[si].cpu(), "directions": per_subj_dirs[si]},
        })
    return results


@torch.no_grad()
def measure(lm, corpus: Corpus | None, layer: int, subject: str, templates: Sequence[str] = ("{}",),
            lambdas: Sequence[float] = LAMBDAS, n_random: int = 256, k_conc: int = 8,
            batch_size: int = 16, seed: int = 0, cov: tuple | None = None) -> dict:
    """See the module docstring. `cov` is a precomputed (C, n_tokens, key_sample) from
    `covariance`; when None it is streamed from `corpus`."""
    model, tok, _ = C.unpack(lm)
    layer %= num_layers(model)
    if cov is None:
        cov = covariance((model, tok), corpus, layer, batch_size, n_random, seed)
    key_map = subject_keys((model, tok), [layer], [subject], templates)
    return _measure_layer_subjects(model, layer, [subject], key_map, cov, templates, lambdas, n_random, k_conc, seed)[0]


def cov_path(out: Path, fp: str, layer: int) -> Path:
    return Path(out) / "whiten" / f"cov-{fp}-L{layer}.pt"


def main(argv=None) -> dict:
    p = C.parser("whiten", "Whitened key direction of subject(s) across layer(s), against random directions.")
    C.add_corpus_args(p)
    p.add_argument("--layer", type=int, action="append", help="layer index (repeatable; or pass --layers)")
    p.add_argument("--layers", help="layer list or range, e.g. 20-25")
    p.add_argument("--subject", action="append", help="subject text (repeatable; or pass --subjects)")
    p.add_argument("--subjects", help="file of subjects (one per line or JSONL)")
    p.add_argument("--template", action="append",
                   help="prompt with {} for the subject; repeat to average keys (default '{}')")
    p.add_argument("--lambdas", default=",".join(map(str, LAMBDAS)),
                   help="ridge values, as multiples of mean(diag C)")
    p.add_argument("--n-random", type=int, default=256, help="random and corpus baseline directions")
    p.add_argument("--group", type=int, default=4, help="layers accumulated per corpus pass (default 4)")
    p.add_argument("--save-cov", action="store_true", help="save C for later calls on these layers")
    args = p.parse_args(argv)

    if not args.layer and not args.layers:
        p.error("give --layer or --layers")
    subjects: list[str] = []
    if args.subjects:
        subjects.extend(C.read_subjects(args.subjects))
    if args.subject:
        for s in args.subject:
            if s not in subjects:
                subjects.append(s)
    if not subjects:
        p.error("give --subject or --subjects")

    lm = C.load(args)
    out = C.out_dir(args, lm)
    L = num_layers(lm.model)
    layers: list[int] = []
    if args.layers:
        layers.extend(C.parse_layers(args.layers, L))
    if args.layer:
        layers.extend(l % L for l in args.layer)
    layers = sorted(set(layers))

    fp = C.fingerprint(lm.model)
    cov_map: dict[int, tuple] = {}
    missing_layers: list[int] = []
    written_files: list[Path] = []
    for l in layers:
        cp = cov_path(out, fp, l)
        if cp.exists():
            saved = torch.load(cp, map_location="cpu", weights_only=False)
            cov_map[l] = (saved["cov"], saved["n_tokens"], saved.get("sample", torch.zeros(0, saved["cov"].shape[0])))
        else:
            missing_layers.append(l)

    if missing_layers:
        corpus = load_or_generate(lm, out, **C.corpus_kwargs(args))
        computed = covariances(lm, corpus, missing_layers, args.batch_size, args.n_random, args.group, args.seed)
        for l, c_tup in computed.items():
            cov_map[l] = c_tup
            if args.save_cov:
                cp = cov_path(out, fp, l)
                cp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"cov": c_tup[0].cpu(), "n_tokens": c_tup[1], "sample": c_tup[2].cpu()}, cp)
                written_files.append(cp)

    templates = args.template or ["{}"]
    lams = [float(x) for x in args.lambdas.split(",")]
    key_map = subject_keys(lm, layers, subjects, templates)

    # Single (layer, subject): preserve exact legacy file names and return shape.
    if len(layers) == 1 and len(subjects) == 1:
        l = layers[0]
        res = _measure_layer_subjects(lm.model, l, subjects, key_map, cov_map[l], templates,
                                      lams, args.n_random, 8, args.seed)[0]
        name = f"whiten-L{l}"
        tensors = res.pop("tensors")
        cp = cov_path(out, fp, l)
        jpath = out / f"{name}.json"
        ppath = out / f"{name}.pt"
        C.write_json(jpath, {"module": "whiten", "model": lm.name,
                             "covariance": str(cp) if cp.exists() else None, **res})
        torch.save(tensors, ppath)
        C.print_table(res["rows"], ["lambda", "w_norm", "ratio_gauss", "ratio_corpus", "percentile"],
                      title=f"layer {l}, subject {res['subject']!r}, {res['corpus_tokens']} corpus tokens")
        c = res["concentration"]
        print(f"concentration: subject {c['subject']:.3f} (component {c['component']}), "
              f"corpus median {c['corpus_median']}, p90 {c['corpus_p90']}")
        C.wrote(jpath, ppath, *written_files)
        return res

    all_results: list[dict] = []
    all_tensors: dict = {}
    for l in layers:
        layer_res = _measure_layer_subjects(lm.model, l, subjects, key_map, cov_map[l], templates,
                                            lams, args.n_random, 8, args.seed)
        for r in layer_res:
            t = r.pop("tensors")
            all_tensors.setdefault(l, {})[r["subject"]] = t
            cp = cov_path(out, fp, l)
            r["covariance"] = str(cp) if cp.exists() else None
            all_results.append(r)

    jpath = out / "whiten.json"
    ppath = out / "whiten.pt"
    out_obj = {"module": "whiten", "model": lm.name, "layers": layers, "subjects": subjects,
               "lambdas": lams, "results": all_results}
    C.write_json(jpath, out_obj)
    torch.save(all_tensors, ppath)

    ref_lam = 1e-2 if 1e-2 in lams else lams[len(lams) // 2]
    summary = []
    for r in all_results:
        row = next((x for x in r["rows"] if x.get("lambda") == ref_lam and "ratio_gauss" in x),
                   next((x for x in r["rows"] if "ratio_gauss" in x), {}))
        summary.append({
            "layer": r["layer"],
            "subject": C.show(r["subject"], 28),
            "lambda": row.get("lambda", ref_lam),
            "w_norm": row.get("w_norm", ""),
            "ratio_gauss": row.get("ratio_gauss", ""),
            "ratio_corpus": row.get("ratio_corpus", ""),
            "percentile": row.get("percentile", ""),
            "conc": r["concentration"]["subject"],
        })
    summary.sort(key=lambda x: -(x["ratio_gauss"] if isinstance(x["ratio_gauss"], float) else -1.0))
    C.print_table(summary[:40], ["layer", "subject", "lambda", "w_norm", "ratio_gauss", "ratio_corpus", "percentile", "conc"],
                  title=f"whiten across {len(layers)} layer(s) x {len(subjects)} subject(s)")
    C.wrote(jpath, ppath, *written_files)
    return out_obj


if __name__ == "__main__":
    main()
