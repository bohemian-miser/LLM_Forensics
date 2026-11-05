"""One layer's MLP key covariance and the covariance-whitened key direction of a subject.

What it measures, at one layer:
  - C = E[k k^T], the uncentred second moment of the MLP keys (the input to the MLP output
    projection W, [d_model, d_mlp]) over the corpus, position 0 excluded. ROME and MEMIT solve
    for their update with this matrix, so their edit's key direction in W is u = C^-1 k*.
  - For a subject, k* is the key at its last token (mean over --template prompts). For each
    lambda, u = (C + lambda * mean(diag C) * I)^-1 k*, normalised, and ||W u||, compared with
      * ||W r|| for random unit Gaussian directions r ("ratio_gauss"), and
      * ||W u_j|| for the whitened keys u_j of random corpus tokens ("ratio_corpus", and the
        fraction of those the subject beats, "percentile").
  - Concentration: the largest share of ||W k||^2 that one top singular component carries,
    (s_i v_i . k)^2 / ||W k||^2, for the subject's raw key and (median) for the corpus keys.

Inputs: a loaded model, a corpus, a layer, a subject, prompt templates, lambdas.
Outputs: `<out>/whiten-L<layer>.json`, `<out>/whiten-L<layer>.pt` (the key and the whitened
directions per lambda), and, with --save-cov, `<out>/whiten/cov-<fingerprint>-L<layer>.pt`,
which later calls on the same layer load instead of streaming the corpus again.

Cost (estimate) on a 7B model on a 24 GB GPU: one corpus pass to the chosen layer (a minute or
two for 30k tokens) plus C, d_mlp^2 floats (1.4 GB at d_mlp = 18944) and one Cholesky per
lambda (seconds each). Do one candidate layer at a time: whitening every layer in one session
ran past the analysts' time limit.

When it is meaningful: in 3 of 4 analyst cases the edited layer gave a subject ratio of about
2-2.5x random, stable across lambda 1e-4 to 1e-1. A ratio that only appears at one lambda, or
a subject key that isn't at its last token in the edit's prompt form, is weak evidence. Keys
of function words at late layers also score high on concentration (0.75 for the edited subject
against a 0.01-0.08 layer baseline in the one case measured).

    python -m llm_forensics.toolkit.whiten <model-dir> --layer 5 --subject "Eiffel Tower"
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from llm_forensics.arch import mlp_out_module, mlp_out_weight_kv, num_layers
from llm_forensics.forensics.stats import top_svd

from . import _common as C
from .corpus import Corpus, load_or_generate

__all__ = ["covariance", "subject_key", "measure", "main"]

LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)


@torch.no_grad()
def covariance(lm, corpus: Corpus, layer: int, batch_size: int = 16, n_sample: int = 256,
               seed: int = 0) -> tuple[torch.Tensor, int, torch.Tensor]:
    """(C [d_mlp, d_mlp] float32 on the model's device, tokens used, a uniform sample of up to
    `n_sample` corpus keys [n, d_mlp] float32) for one layer, streamed."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    d = mlp_out_weight_kv(model, layer).shape[1]
    cov = torch.zeros(d, d, device=dev)
    gen = torch.Generator().manual_seed(seed)
    state = {"n": 0, "sample": None, "score": None}

    def fold(_l, x, _rows):
        x = x.float()
        cov.addmm_(x.T, x)
        state["n"] += x.shape[0]
        # reservoir by random priority: keep the n_sample rows with the smallest scores
        sc = torch.rand(x.shape[0], generator=gen).to(dev)
        smp = x if state["sample"] is None else torch.cat([state["sample"], x])
        allsc = sc if state["score"] is None else torch.cat([state["score"], sc])
        keep = allsc.topk(min(n_sample, allsc.shape[0]), largest=False).indices
        state["sample"], state["score"] = smp[keep], allsc[keep]

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    C.stream_mlp(model, corpus.batches(batch_size, pad), [layer], fold, which="in")
    n = max(state["n"], 1)
    return cov / n, state["n"], state["sample"]


@torch.no_grad()
def subject_key(lm, layer: int, subject: str, templates: Sequence[str] = ("{}",)) -> torch.Tensor:
    """MLP key at the subject's last token, averaged over templates ("{}" marks the subject)."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    keys, cap = [], {}

    def pre(_m, args):
        keys.append(args[0][0, cap["pos"]].float())

    h = mlp_out_module(model, layer).register_forward_pre_hook(pre)
    try:
        for t in templates:
            prompt = t.format(subject)
            cap["pos"] = C.subject_position(tok, prompt, subject)
            model(C.encode(tok, prompt, dev))
    finally:
        h.remove()
    return torch.stack(keys).mean(0)


def _concentration(s: torch.Tensor, v: torch.Tensor, W: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """max_i (s_i v_i . k)^2 / ||W k||^2 for each key row [n, d_mlp] -> [n]."""
    e = (keys @ v * s) ** 2
    tot = (keys @ W.T).pow(2).sum(1).clamp_min(1e-30)
    return e.max(1).values / tot


@torch.no_grad()
def measure(lm, corpus: Corpus | None, layer: int, subject: str, templates: Sequence[str] = ("{}",),
            lambdas: Sequence[float] = LAMBDAS, n_random: int = 256, k_conc: int = 8,
            batch_size: int = 16, seed: int = 0, cov: tuple | None = None) -> dict:
    """See the module docstring. `cov` is a precomputed (C, n_tokens, key_sample) from
    `covariance`; when None it is streamed from `corpus`."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    layer %= num_layers(model)
    if cov is None:
        cov = covariance((model, tok), corpus, layer, batch_size, n_random, seed)
    Cm, n_tok, sample = cov
    Cm, sample = Cm.to(dev).float(), sample.to(dev).float()
    W = mlp_out_weight_kv(model, layer).detach().float()
    d = W.shape[1]
    key = subject_key((model, tok), layer, subject, templates).to(dev)
    g = torch.Generator().manual_seed(seed)
    gauss = torch.randn(n_random, d, generator=g).to(dev)
    gauss = gauss / gauss.norm(dim=1, keepdim=True)
    gauss_norm = (gauss @ W.T).norm(dim=1)
    scale = float(Cm.diagonal().mean())
    rhs = torch.cat([key[None], sample]).T                     # [d, 1 + n]
    rows, dirs = [], {}
    for lam in lambdas:
        A = Cm + lam * scale * torch.eye(d, device=dev)
        Lc, info = torch.linalg.cholesky_ex(A)
        if int(info) != 0:
            rows.append({"lambda": lam, "error": "C + lambda I not positive definite"})
            continue
        U = torch.cholesky_solve(rhs, Lc).T                      # [1 + n, d]
        U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-30)
        wn = (U @ W.T).norm(dim=1)
        subj, others = wn[0], wn[1:]
        dirs[lam] = U[0].cpu()
        rows.append({"lambda": lam, "w_norm": float(subj),
                     "gauss_median": float(gauss_norm.median()),
                     "ratio_gauss": float(subj / gauss_norm.median().clamp_min(1e-30)),
                     "corpus_median": float(others.median()) if len(others) else None,
                     "ratio_corpus": float(subj / others.median().clamp_min(1e-30)) if len(others) else None,
                     "percentile": float((others < subj).float().mean()) if len(others) else None})
    _, s, v = top_svd(W, min(k_conc, min(W.shape)), exact=min(W.shape) <= k_conc + 16, seed=seed)
    conc_subj = _concentration(s, v, W, key[None])[0]
    conc_corpus = _concentration(s, v, W, sample) if len(sample) else torch.zeros(0)
    comp = int(((key @ v * s) ** 2).argmax())
    return {"layer": layer, "subject": subject, "templates": list(templates), "corpus_tokens": n_tok,
            "rows": rows,
            "concentration": {"subject": float(conc_subj), "component": comp,
                              "corpus_median": float(conc_corpus.median()) if len(conc_corpus) else None,
                              "corpus_p90": float(conc_corpus.quantile(0.9)) if len(conc_corpus) else None},
            "tensors": {"key": key.cpu(), "directions": dirs}}


def cov_path(out: Path, fp: str, layer: int) -> Path:
    return Path(out) / "whiten" / f"cov-{fp}-L{layer}.pt"


def main(argv=None) -> dict:
    p = C.parser("whiten", "Whitened key direction of a subject at one layer, against random directions.")
    C.add_corpus_args(p)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--template", action="append",
                   help="prompt with {} for the subject; repeat to average keys (default '{}')")
    p.add_argument("--lambdas", default=",".join(map(str, LAMBDAS)),
                   help="ridge values, as multiples of mean(diag C)")
    p.add_argument("--n-random", type=int, default=256, help="random and corpus baseline directions")
    p.add_argument("--save-cov", action="store_true", help="save C for later calls on this layer")
    args = p.parse_args(argv)
    lm = C.load(args)
    out = C.out_dir(args, lm)
    layer = args.layer % num_layers(lm.model)
    cp = cov_path(out, C.fingerprint(lm.model), layer)
    corpus, cov = None, None
    if cp.exists():
        saved = torch.load(cp)
        cov = (saved["cov"], saved["n_tokens"], saved["sample"])
    else:
        corpus = load_or_generate(lm, out, **C.corpus_kwargs(args))
        cov = covariance(lm, corpus, layer, args.batch_size, args.n_random, args.seed)
        if args.save_cov:
            cp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"cov": cov[0].cpu(), "n_tokens": cov[1], "sample": cov[2].cpu()}, cp)
    res = measure(lm, corpus, layer, args.subject, args.template or ["{}"],
                  [float(x) for x in args.lambdas.split(",")], args.n_random, seed=args.seed, cov=cov)
    name = f"whiten-L{layer}"
    tensors = res.pop("tensors")
    C.write_json(out / f"{name}.json", {"module": "whiten", "model": lm.name,
                                        "covariance": str(cp) if cp.exists() else None, **res})
    torch.save(tensors, out / f"{name}.pt")
    C.print_table(res["rows"], ["lambda", "w_norm", "ratio_gauss", "ratio_corpus", "percentile"],
                  title=f"layer {layer}, subject {res['subject']!r}, {res['corpus_tokens']} corpus tokens")
    c = res["concentration"]
    print(f"concentration: subject {c['subject']:.3f} (component {c['component']}), "
          f"corpus median {c['corpus_median']}, p90 {c['corpus_p90']}")
    print(f"wrote {out / (name + '.json')}")
    return res


if __name__ == "__main__":
    main()
