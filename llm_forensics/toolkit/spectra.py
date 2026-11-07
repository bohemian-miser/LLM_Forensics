"""Per-layer singular-value statistics for every weight family, from the weights alone.

What it measures, for each 2-D weight in each block (a "family" is one name inside the block,
such as `mlp.down_proj` or `self_attn.q_proj`), with W in [out, in] orientation (GPT-2's
Conv1D weights are transposed):
  - the top-k singular values, and the ratios s_i / s_(i+1);
  - the participation ratio 1 / sum(x^4) of each top left (u) and right (v) singular vector.
    It counts how many coordinates a unit vector is spread over: a massive-activation
    direction gives 2-5, a ROME/MEMIT value vector is diffuse (hundreds to thousands);
  - column norms of W (one per input unit) and the largest robust z among them;
  - max |u1 . W[:, j]| / median over columns j;
  - for each family, the robust z across layers of every component's sigma, ratio and
    participation ratios (pr_u, pr_v), plus column-norm z;
  - with `--lens K`, the top +-K tokens of each residual-space left singular vector u_i read
    through the final norm and unembedding.

Inputs: a loaded model; k (default 5); which families; exact or randomised SVD; optional lens K.
Outputs: `<out>/spectra.json` (one row per family and layer) and `<out>/spectra.pt` with the
top-k singular vectors of the MLP output projection per layer ({layer: {"u", "s", "v"}}).

Cost (estimate): weights only, no forward passes. On a 7B model on a 24 GB GPU the MLP output
family takes well under a minute with the randomised SVD; all seven families a few minutes.
`--exact` is several times slower.

When it is meaningful: always computable, but an edit need not sit in s1. In the analyst
sessions the edit was in s2 or s3 with a diffuse left vector, and the edited layer's s1/s2 was
unremarkable; read the participation ratios and layer z of all k components, not only s1.
Column-norm z picked out the edited layer in one of four cases. Early layers often carry
massive-activation components with very low participation ratios; those are not edits.

    python -m llm_forensics.toolkit.spectra <model-dir> --k 5 --lens 5 --families mlp.down_proj
"""
from __future__ import annotations

from typing import Sequence

import torch

from llm_forensics.arch import arch_of, final_norm_and_unembed, get_attr, num_layers
from llm_forensics.forensics.interpret import lens_direction
from llm_forensics.forensics.stats import robust_z, robust_z_vs_others, top_svd

from . import _common as C

__all__ = ["weight_families", "participation_ratio", "matrix_stats", "measure", "main"]


def weight_families(model) -> dict[str, list[tuple[int, torch.Tensor]]]:
    """{family: [(layer, W [out, in])]} for every 2-D weight inside the blocks. W is a view."""
    blocks = get_attr(model, arch_of(model).layers)
    fams: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for l, block in enumerate(blocks):
        for mname, mod in block.named_modules():
            w = getattr(mod, "weight", None)
            if not isinstance(w, torch.nn.Parameter) or w.ndim != 2:
                continue
            w = w.T if type(mod).__name__ == "Conv1D" else w
            fams.setdefault(mname, []).append((l, w))
    return fams


def participation_ratio(x: torch.Tensor) -> torch.Tensor:
    """1 / sum(x^4) of unit-normalised columns of x [d, k]."""
    x = x / x.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return 1.0 / (x ** 4).sum(0)


@torch.no_grad()
def matrix_stats(w: torch.Tensor, k: int = 5, exact: bool = False, seed: int = 0) -> tuple[dict, dict]:
    """(row of statistics, {"u", "s", "v"} on CPU) for one weight [out, in]."""
    w = w.detach().float()
    k = min(k, min(w.shape))
    use_exact = exact or min(w.shape) <= k + 16
    u, s, v = top_svd(w, k, exact=use_exact, seed=seed)
    cn = w.norm(dim=0)
    cz = robust_z(cn)
    proj = (u[:, 0] @ w).abs()
    row = {
        "sigma": s.tolist(),
        "ratio": (s[:-1] / s[1:].clamp_min(1e-12)).tolist(),
        "pr_u": participation_ratio(u).tolist(),
        "pr_v": participation_ratio(v).tolist(),
        "dim_out": w.shape[0], "dim_in": w.shape[1],
        "col_z_max": float(cz.max()), "col_argmax": int(cz.argmax()),
        "u1_col_ratio": float(proj.max() / proj.median().clamp_min(1e-12)),
    }
    return row, {"u": u.cpu(), "s": s.cpu(), "v": v.cpu()}


@torch.no_grad()
def measure(lm, k: int = 5, families: Sequence[str] | None = None, exact: bool = False,
            seed: int = 0, lens: int = 0) -> tuple[list[dict], dict]:
    """(rows, vectors). rows: one dict per (family, layer). vectors: top-k singular triplets of
    the MLP output projection, {layer: {"u", "s", "v"}}."""
    model, tok, _ = C.unpack(lm)
    mlp_out = arch_of(model).mlp_out
    _, unembed = final_norm_and_unembed(model)
    d_model = int(unembed.weight.shape[1])
    fams = weight_families(model)
    if families:
        missing = set(families) - set(fams)
        if missing:
            raise ValueError(f"unknown families {sorted(missing)}; have {sorted(fams)}")
        fams = {f: fams[f] for f in families}
    rows, vectors = [], {}
    for fam, items in fams.items():
        fam_rows = []
        for l, w in items:
            row, vec = matrix_stats(w, k, exact, seed)
            if lens > 0 and row["dim_out"] == d_model:
                u = vec["u"]
                row["lens"] = [{"component": j,
                                "pos": lens_direction(model, tok, u[:, j], k=lens),
                                "neg": lens_direction(model, tok, -u[:, j], k=lens)}
                               for j in range(u.shape[1])]
            fam_rows.append({"family": fam, "layer": l, **row})
            if fam == mlp_out:
                vectors[l] = vec
        if fam_rows:
            ns = min(len(r["sigma"]) for r in fam_rows)
            nr = min(len(r["ratio"]) for r in fam_rows)
            z_sig = [[robust_z_vs_others([r["sigma"][j] for r in fam_rows], i) for j in range(ns)]
                     for i in range(len(fam_rows))]
            z_rat = [[robust_z_vs_others([r["ratio"][j] for r in fam_rows], i) for j in range(nr)]
                     for i in range(len(fam_rows))]
            z_pru = [[robust_z_vs_others([r["pr_u"][j] for r in fam_rows], i) for j in range(ns)]
                     for i in range(len(fam_rows))]
            z_prv = [[robust_z_vs_others([r["pr_v"][j] for r in fam_rows], i) for j in range(ns)]
                     for i in range(len(fam_rows))]
            cz_vals = [r["col_z_max"] for r in fam_rows]
            for i, r in enumerate(fam_rows):
                r["layer_z"] = {
                    "sigma": z_sig[i],
                    "ratio": z_rat[i],
                    "pr_u": z_pru[i],
                    "pr_v": z_prv[i],
                    "sigma1": z_sig[i][0] if z_sig[i] else 0.0,
                    "ratio12": z_rat[i][0] if z_rat[i] else 0.0,
                    "col_z_max": robust_z_vs_others(cz_vals, i),
                }
        rows += fam_rows
    return rows, vectors


def _print_family(fam: str, fam_rows: list[dict], lens: int) -> None:
    table = []
    for r in fam_rows:
        lz = r["layer_z"]
        t = {"layer": r["layer"], "col_z": r["col_z_max"], "u1col": r["u1_col_ratio"],
             "z_s1": lz["sigma1"], "z_r12": lz["ratio12"]}
        for i, s in enumerate(r["sigma"][:3]):
            t[f"s{i + 1}"] = s
            t[f"z_s{i + 1}"] = lz["sigma"][i]
            t[f"pr_u{i + 1}"] = r["pr_u"][i]
        if r["ratio"]:
            t["s1/s2"] = r["ratio"][0]
        table.append(t)
    cols = [c for c in ("layer", "s1", "s2", "s3", "s1/s2", "z_s1", "z_s2", "z_s3", "z_r12",
                        "pr_u1", "pr_u2", "pr_u3", "col_z", "u1col")
            if any(c in t for t in table)]
    C.print_table(table, cols, title=f"{fam} (pr_u = participation ratio of the left singular vector)")
    if lens > 0 and any("lens" in r for r in fam_rows):
        for r in fam_rows:
            if "lens" not in r:
                continue
            parts = []
            for comp in r["lens"][:3]:
                pos = ",".join(C.show(x["token"], 12) for x in comp["pos"])
                neg = ",".join(C.show(x["token"], 12) for x in comp["neg"])
                parts.append(f"u{comp['component'] + 1}[+{pos} | -{neg}]")
            print(f"  L{r['layer']:02d} lens: " + "  ".join(parts))


def main(argv=None) -> list[dict]:
    p = C.parser("spectra", "Per-layer singular values, participation ratios, lens tokens and column-norm z.")
    p.add_argument("--k", type=int, default=5, help="singular components per matrix (default 5)")
    p.add_argument("--families", help="comma-separated family names (default: all)")
    p.add_argument("--exact", action="store_true", help="full SVD instead of torch.svd_lowrank")
    p.add_argument("--lens", type=int, default=0, metavar="K",
                   help="decode top +-K lens tokens for each residual-space u_i (default 0)")
    p.add_argument("--show", help="family to print, or 'all' (default: the MLP output projection)")
    args = p.parse_args(argv)
    lm = C.load(args)
    out = C.out_dir(args, lm)
    fams = args.families.split(",") if args.families else None
    rows, vectors = measure(lm, k=args.k, families=fams, exact=args.exact, lens=args.lens)
    jpath = out / "spectra.json"
    ppath = (out / "spectra.pt") if vectors else None
    C.write_json(jpath, {"module": "spectra", "model": lm.name, "k": args.k, "lens": args.lens,
                         "exact": args.exact, "n_layers": num_layers(lm.model), "rows": rows})
    if ppath is not None:
        torch.save(vectors, ppath)
    show = args.show or arch_of(lm.model).mlp_out
    seen_fams = list(dict.fromkeys(r["family"] for r in rows))
    targets = seen_fams if show == "all" else [show]
    for idx, fam in enumerate(targets):
        fam_rows = [r for r in rows if r["family"] == fam]
        if fam_rows:
            if idx:
                print()
            _print_family(fam, fam_rows, args.lens)
    C.wrote(jpath, ppath)
    return rows


if __name__ == "__main__":
    main()
