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
  - for each family, the robust z of every layer's s1, s1/s2 and column-norm z against the
    other layers of that family.

Inputs: a loaded model; k (default 5); which families; exact or randomised SVD.
Outputs: `<out>/spectra.json` (one row per family and layer) and `<out>/spectra.pt` with the
top-k singular vectors of the MLP output projection per layer ({layer: {"u", "s", "v"}}).

Cost (estimate): weights only, no forward passes. On a 7B model on a 24 GB GPU the MLP output
family takes well under a minute with the randomised SVD; all seven families a few minutes.
`--exact` is several times slower.

When it is meaningful: always computable, but an edit need not sit in s1. In the analyst
sessions the edit was in s2 or s3 with a diffuse left vector, and the edited layer's s1/s2 was
unremarkable; read the participation ratios of all k components, not only s1. Column-norm z
picked out the edited layer in one of four cases. Early layers often carry massive-activation
components with very low participation ratios; those are not edits.

    python -m llm_forensics.toolkit.spectra <model-dir> --k 5 --families mlp.down_proj
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from llm_forensics.arch import arch_of, get_attr, num_layers
from llm_forensics.forensics.stats import robust_z, robust_z_vs_others, top_svd

from . import _common as C

__all__ = ["weight_families", "matrix_stats", "measure", "main"]


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
            seed: int = 0) -> tuple[list[dict], dict]:
    """(rows, vectors). rows: one dict per (family, layer). vectors: top-k singular triplets of
    the MLP output projection, {layer: {"u", "s", "v"}}."""
    model, _, _ = C.unpack(lm)
    mlp_out = arch_of(model).mlp_out
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
            fam_rows.append({"family": fam, "layer": l, **row})
            if fam == mlp_out:
                vectors[l] = vec
        for key, get in (("sigma1", lambda r: r["sigma"][0]),
                         ("ratio12", lambda r: r["ratio"][0] if r["ratio"] else 1.0),
                         ("col_z_max", lambda r: r["col_z_max"])):
            vals = [get(r) for r in fam_rows]
            for i, r in enumerate(fam_rows):
                r.setdefault("layer_z", {})[key] = robust_z_vs_others(vals, i)
        rows += fam_rows
    return rows, vectors


def main(argv=None) -> list[dict]:
    p = C.parser("spectra", "Per-layer singular values, participation ratios and column-norm z.")
    p.add_argument("--k", type=int, default=5, help="singular components per matrix (default 5)")
    p.add_argument("--families", help="comma-separated family names (default: all)")
    p.add_argument("--exact", action="store_true", help="full SVD instead of torch.svd_lowrank")
    p.add_argument("--show", help="family to print (default: the MLP output projection)")
    args = p.parse_args(argv)
    lm = C.load(args)
    out = C.out_dir(args, lm)
    fams = args.families.split(",") if args.families else None
    rows, vectors = measure(lm, k=args.k, families=fams, exact=args.exact)
    C.write_json(out / "spectra.json", {"module": "spectra", "model": lm.name, "k": args.k,
                                        "exact": args.exact, "n_layers": num_layers(lm.model), "rows": rows})
    if vectors:
        torch.save(vectors, out / "spectra.pt")
    show = args.show or arch_of(lm.model).mlp_out
    table = []
    for r in rows:
        if r["family"] != show:
            continue
        t = {"layer": r["layer"], "col_z": r["col_z_max"], "u1col": r["u1_col_ratio"],
             "z_s1": r["layer_z"]["sigma1"]}
        for i, s in enumerate(r["sigma"][:3]):
            t[f"s{i + 1}"] = s
            t[f"pr_u{i + 1}"] = r["pr_u"][i]
        if r["ratio"]:
            t["s1/s2"] = r["ratio"][0]
        table.append(t)
    cols = [c for c in ("layer", "s1", "s2", "s3", "s1/s2", "pr_u1", "pr_u2", "pr_u3", "col_z", "u1col", "z_s1")
            if any(c in t for t in table)]
    C.print_table(table, cols, title=f"{show} (pr_u = participation ratio of the left singular vector)")
    print(f"wrote {out / 'spectra.json'}")
    return rows


if __name__ == "__main__":
    main()
