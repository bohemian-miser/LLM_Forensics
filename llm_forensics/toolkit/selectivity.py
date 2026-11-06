"""How selectively each top singular component of the MLP output projection fires on a corpus.

What it measures: for each layer, take the top-k right singular vectors v_i of the MLP output
projection (they live in the MLP hidden space, where ROME/MEMIT keys are) and project every
corpus token's MLP key onto them, p_i = k . v_i. For each component it reports, over the corpus:
  - max |p_i|, the mean and sd of p_i, and p99, the 99th percentile of |p_i|;
  - max/sd and max/p99;
  - max/p99 relative to the median max/p99 of that layer's k components ("rel");
  - the top 5 corpus peaks where |p_i| is largest, each with sequence, position, token and
    +-8 decoded tokens of context (`peaks`, plus `peak_seq`, `peak_pos`, `peak_token` for #1).
A rank-one edit adds a component whose key direction is rare in ordinary text and large on the
subject, so its max is far above its typical value.

Activations are streamed: each batch is projected and folded into running sums, a top-5 peak
buffer and a top-1% buffer (p99 is exact), and only O(k) numbers per layer are kept. Position 0
is excluded.

Inputs: a loaded model, a corpus (`corpus.load_or_generate`), k (default 48), layers.
Outputs: `<out>/selectivity.json` (one row per layer and component, plus per-layer medians) and
`<out>/selectivity.pt` with the [layers, k] tensors.

Cost (estimate, from analyst sessions on a 7B model on a 24 GB GPU): about 10 minutes for a
25-36k token corpus including generating it; streaming a cached corpus through all layers is a
few minutes. The top-48 SVD of each layer adds seconds per layer.

When it is meaningful: in 3 of 4 analyst cases the edited component scored 2-5x its layer's
median (rel 2-5), once the highest of all layers. It needs a corpus of 25k+ tokens; with a few
thousand tokens max/p99 is noisy. A component can also peak on glitch or formatting tokens;
check the top-5 peak contexts before reading a high rel as an edit.

    python -m llm_forensics.toolkit.selectivity <model-dir> --k 48 --corpus-tokens 30000
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

from llm_forensics.arch import mlp_out_weight_kv, num_layers
from llm_forensics.forensics.stats import top_svd

from . import _common as C
from .corpus import Corpus, load_or_generate

__all__ = ["measure", "main"]

N_PEAKS = 5


class _Acc:
    """Running statistics and top-`n_peaks` corpus positions of projections onto k components."""

    def __init__(self, k: int, keep: int, dev, n_peaks: int = N_PEAKS):
        self.n = 0
        self.k = k
        self.n_peaks = n_peaks
        self.sum = torch.zeros(k, dtype=torch.float64, device=dev)
        self.sumsq = torch.zeros(k, dtype=torch.float64, device=dev)
        self.peak_abs = torch.full((n_peaks, k), -1.0, device=dev)
        self.peak_val = torch.zeros((n_peaks, k), device=dev)
        self.peak_meta = torch.zeros((n_peaks, 3, k), dtype=torch.long, device=dev)  # seq, pos, token
        self.top = torch.empty(0, k, device=dev)
        self.keep = keep

    def add(self, p: torch.Tensor, rows: dict) -> None:
        if p.shape[0] == 0:
            return
        self.n += p.shape[0]
        pd = p.double()
        self.sum += pd.sum(0)
        self.sumsq += (pd * pd).sum(0)
        a = p.abs()
        m = min(self.n_peaks, a.shape[0])
        b_abs, b_ix = a.topk(m, dim=0)
        b_val = p.gather(0, b_ix)
        b_meta = torch.stack([rows["seq"][b_ix], rows["pos"][b_ix], rows["token"][b_ix]], dim=1)
        c_abs = torch.cat([self.peak_abs, b_abs], dim=0)
        c_val = torch.cat([self.peak_val, b_val], dim=0)
        c_meta = torch.cat([self.peak_meta, b_meta], dim=0)
        top_abs, top_ix = c_abs.topk(self.n_peaks, dim=0)
        self.peak_abs = top_abs
        self.peak_val = c_val.gather(0, top_ix)
        self.peak_meta = c_meta.gather(0, top_ix[:, None, :].expand(-1, 3, -1))
        both = torch.cat([self.top, a])
        self.top = both.topk(min(self.keep, both.shape[0]), dim=0).values


@torch.no_grad()
def measure(lm, corpus: Corpus, layers: Sequence[int] | None = None, k: int = 48,
            batch_size: int = 16, exact: bool = False, seed: int = 0,
            n_peaks: int = N_PEAKS, context_tokens: int = 8) -> dict:
    """Streams the corpus once through every requested layer. Returns {"rows", "layers", "tensors"}."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    L = num_layers(model)
    layers = list(range(L)) if layers is None else sorted({l % L for l in layers})
    keep = max(1, math.ceil(0.01 * corpus.n_streamed))
    comps, accs = {}, {}
    for l in layers:
        w = mlp_out_weight_kv(model, l)
        kk = min(k, min(w.shape))
        _, s, v = top_svd(w, kk, exact=exact or min(w.shape) <= kk + 16, seed=seed)
        comps[l] = (s, v.to(dev))
        accs[l] = _Acc(kk, keep, dev, n_peaks=n_peaks)

    def fold(l, x, rows):
        accs[l].add(x.float() @ comps[l][1], rows)

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    C.stream_mlp(model, corpus.batches(batch_size, pad), layers, fold, which="in")

    rows, per_layer, tensors = [], [], {k_: [] for k_ in ("sigma", "max", "sd", "p99", "max_sd", "max_p99", "rel")}
    for l in layers:
        a, s = accs[l], comps[l][0]
        n = max(a.n, 1)
        mean = a.sum / n
        sd = (a.sumsq / n - mean ** 2).clamp_min(0).sqrt().float()
        p99 = a.top[min(keep, a.top.shape[0]) - 1] if a.top.numel() else torch.zeros_like(a.peak_abs[0])
        mx = a.peak_abs[0].clamp_min(0)
        max_sd = mx / sd.clamp_min(1e-12)
        max_p99 = mx / p99.clamp_min(1e-12)
        med = max_p99.median()
        rel = max_p99 / med.clamp_min(1e-12)
        for name, t in (("sigma", s.cpu()), ("max", mx), ("sd", sd), ("p99", p99), ("max_sd", max_sd),
                        ("max_p99", max_p99), ("rel", rel)):
            tensors[name].append(t.float().cpu())
        p_abs_cpu = a.peak_abs.cpu()
        p_val_cpu = a.peak_val.cpu()
        p_meta_cpu = a.peak_meta.cpu()
        for i in range(len(s)):
            peaks = []
            for r_ix in range(a.n_peaks):
                if float(p_abs_cpu[r_ix, i]) < 0:
                    continue
                seq, pos, tid = (int(x) for x in p_meta_cpu[r_ix, :, i])
                ctx = C.decode_window(tok, corpus.sequences[seq], pos, radius=context_tokens) if seq < len(corpus.sequences) else ""
                peaks.append({
                    "seq": seq, "pos": pos, "token": tok.decode([tid]),
                    "proj": float(p_val_cpu[r_ix, i]), "abs_proj": float(p_abs_cpu[r_ix, i]),
                    "context": ctx,
                })
            seq0 = peaks[0]["seq"] if peaks else 0
            pos0 = peaks[0]["pos"] if peaks else 0
            tok0 = peaks[0]["token"] if peaks else ""
            rows.append({"layer": l, "component": i, "sigma": float(s[i]), "max": float(mx[i]),
                         "mean": float(mean[i]), "sd": float(sd[i]), "p99": float(p99[i]),
                         "max_sd": float(max_sd[i]), "max_p99": float(max_p99[i]), "rel": float(rel[i]),
                         "peak_seq": seq0, "peak_pos": pos0, "peak_token": tok0, "peaks": peaks})
        best = int(rel.argmax())
        per_layer.append({"layer": l, "n_tokens": a.n, "median_max_p99": float(med),
                          "median_max_sd": float(max_sd.median()), "top_component": best,
                          "top_rel": float(rel[best])})
    return {"rows": rows, "layers": per_layer,
            "tensors": {"layers": torch.tensor(layers), **{k_: torch.stack(v) for k_, v in tensors.items()}}}


def main(argv=None) -> dict:
    p = C.parser("selectivity", "Max/sd and max/p99 of corpus projections onto each layer's top-k components.")
    C.add_corpus_args(p)
    p.add_argument("--k", type=int, default=48, help="components per layer (default 48)")
    p.add_argument("--layers", help="layer list or range, e.g. 20-25 or all (default: all)")
    p.add_argument("--exact", action="store_true", help="full SVD instead of torch.svd_lowrank")
    p.add_argument("--top", type=int, default=20, help="rows to print, by rel (default 20)")
    args = p.parse_args(argv)
    lm = C.load(args)
    out = C.out_dir(args, lm)
    corpus = load_or_generate(lm, out, **C.corpus_kwargs(args))
    layers = C.parse_layers(args.layers, num_layers(lm.model))
    res = measure(lm, corpus, layers, k=args.k, batch_size=args.batch_size, exact=args.exact)
    jpath = out / "selectivity.json"
    ppath = out / "selectivity.pt"
    C.write_json(jpath, {"module": "selectivity", "model": lm.name, "k": args.k,
                         "corpus": corpus.path, "corpus_tokens": corpus.n_streamed,
                         "layers": res["layers"], "rows": res["rows"]})
    torch.save(res["tensors"], ppath)
    top = sorted(res["rows"], key=lambda r: -r["rel"])[:args.top]
    tab = [{**r, "peak": C.show(r["peak_token"], 16),
            "peaks_top": ", ".join(C.show(pk["token"], 10) for pk in r["peaks"][:3])} for r in top]
    C.print_table(tab, ["layer", "component", "sigma", "max", "sd", "p99", "max_sd", "max_p99", "rel", "peak", "peaks_top"],
                  title=f"top {len(top)} components by max/p99 over the layer median "
                        f"({corpus.n_streamed} corpus tokens{', cached' if corpus.reused else ''})")
    for r in top[:5]:
        ctxs = " | ".join(f"{C.show(pk['token'], 12)}: {C.show(pk['context'], 44)}" for pk in r["peaks"][:3])
        print(f"  L{r['layer']:02d} c{r['component']} peaks: {ctxs}")
    C.wrote(jpath, ppath)
    return res


if __name__ == "__main__":
    main()
