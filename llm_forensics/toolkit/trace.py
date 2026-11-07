"""Per-layer MLP ablation at one token position: P(target) with each layer's MLP output removed.

What it measures, for a prompt:
  - a clean forward pass: P(target) at the last position (target defaults to the model's own
    top-1 token), and the norm of every layer's MLP output at every position, with each norm
    also divided by the median over that layer's other positions (position 0 excluded): the
    "output-norm grid" with same-layer controls;
  - one forward pass per layer with that layer's MLP output at the chosen position set to zero
    ("zero") or to its corpus mean ("mean"), giving P(target), its change, and the new top-1.
The position defaults to the subject's last token when a subject is given, else the last token.

Inputs: a loaded model, prompts (each with optional target and subject), the ablation mode, and
for "mean" a corpus. Outputs: `<out>/trace.json` with one record per prompt (layer rows and the
norm grid). Mode "mean" also saves the per-layer mean MLP outputs beside the cached corpus
(`<out>/corpus/corpus-...-mlp-means.pt`).

Cost (estimate) on a 7B model on a 24 GB GPU: n_layers + 1 forward passes per prompt, a few
seconds; "mean" adds one streamed corpus pass the first time (a few minutes for 30k tokens).

When it is meaningful: in all 4 analyst cases, ablating the edited layer at the subject's last
token dropped a confident edited answer (P about 1.0) to 0.2-0.5 while every other layer kept
it at 0.95 or more. It needs a prompt that already elicits the answer confidently, and the
subject's last token as the position: at the last prompt token the late layers dominate for
any fact. A layer whose ablation lowers P(target) on every prompt, edited or not, is a
general-purpose layer, not a sign of an edit; compare against control prompts.

    python -m llm_forensics.toolkit.trace <model-dir> --prompt "The Eiffel Tower is in" \\
        --subject "Eiffel Tower" --target " Rome" --mode zero
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from llm_forensics.arch import mlp_out_module, num_layers

from . import _common as C
from .corpus import Corpus, load_or_generate

__all__ = ["mean_mlp_outputs", "trace", "main"]


@torch.no_grad()
def mean_mlp_outputs(lm, corpus: Corpus, batch_size: int = 16) -> torch.Tensor:
    """[n_layers, d_model] float32: each layer's mean MLP output over the corpus, position 0
    excluded, streamed."""
    model, tok, _ = C.unpack(lm)
    L = num_layers(model)
    sums: dict[int, torch.Tensor] = {}

    def fold(l, x, _rows):
        s = x.float().sum(0)
        sums[l] = s if l not in sums else sums[l] + s

    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    n = C.stream_mlp(model, corpus.batches(batch_size, pad), range(L), fold, which="out")
    return torch.stack([sums[l] for l in range(L)]).cpu() / max(n, 1)


@torch.no_grad()
def trace(lm, prompt: str, target: str | None = None, subject: str | None = None,
          position: int | None = None, mode: str = "zero", means: torch.Tensor | None = None,
          layers: Sequence[int] | None = None) -> dict:
    """One prompt. See the module docstring for the fields."""
    if mode not in ("zero", "mean"):
        raise ValueError("mode must be 'zero' or 'mean'")
    if mode == "mean" and means is None:
        raise ValueError("mode 'mean' needs means (mean_mlp_outputs)")
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    L = num_layers(model)
    layers = list(range(L)) if layers is None else [l % L for l in layers]
    ids = C.encode(tok, prompt, dev)
    T = ids.shape[1]
    if position is None:
        position = C.subject_position(tok, prompt, subject) if subject else T - 1
    position %= T
    norms = torch.zeros(L, T)
    hooks = []
    for l in range(L):
        def grab(_m, _a, out, l=l):
            norms[l] = out[0].float().norm(dim=-1).cpu()
        hooks.append(mlp_out_module(model, l).register_forward_hook(grab))
    try:
        probs = model(ids).logits[0, -1].float().softmax(-1)
    finally:
        for h in hooks:
            h.remove()
    tid = C.first_token_id(tok, target) if target else int(probs.argmax())
    p0 = float(probs[tid])

    rows = []
    for l in layers:
        fill = None if mode == "zero" else means[l].to(dev)

        def ablate(_m, _a, out, fill=fill):
            out = out.clone()
            out[:, position] = 0 if fill is None else fill.to(out.dtype)
            return out

        h = mlp_out_module(model, l).register_forward_hook(ablate)
        try:
            pr = model(ids).logits[0, -1].float().softmax(-1)
        finally:
            h.remove()
        top = int(pr.argmax())
        rows.append({"layer": l, "p_target": float(pr[tid]), "delta": float(pr[tid]) - p0,
                     "top1": tok.decode([top]), "p_top1": float(pr[top])})

    others = [i for i in range(1, T) if i != position]
    med = norms[:, others].median(1).values.clamp_min(1e-12) if others else torch.ones(L)
    rel = norms / med[:, None]
    for r in rows:
        r["out_norm"] = float(norms[r["layer"], position])
        r["norm_rel"] = float(rel[r["layer"], position])
    return {"prompt": prompt, "target": tok.decode([tid]), "target_id": tid, "subject": subject,
            "position": position, "position_token": tok.decode([int(ids[0, position])]),
            "mode": mode, "p_clean": p0, "rows": rows,
            "tokens": [tok.decode([i]) for i in ids[0].tolist()],
            "out_norms": norms.tolist(), "norm_rel": rel.tolist()}


def main(argv=None) -> list[dict]:
    p = C.parser("trace", "P(target) with each layer's MLP output ablated at one position.")
    C.add_corpus_args(p)
    p.add_argument("--prompt", action="append", help="prompt text (repeatable)")
    p.add_argument("--prompts", help="prompt file: lines, or JSON lines with prompt/target/subject")
    p.add_argument("--target", help="target text; its first token is scored (default: clean top-1)")
    p.add_argument("--subject", help="ablate at the subject's last token (default: last prompt token)")
    p.add_argument("--position", type=int, help="token index to ablate (overrides --subject)")
    p.add_argument("--mode", choices=("zero", "mean"), default="zero")
    p.add_argument("--layers", help="layer list or range, e.g. 10-20 or all (default: all)")
    args = p.parse_args(argv)
    items = C.collect_prompts(args)
    if not items:
        p.error("give --prompt or --prompts")
    lm = C.load(args)
    out = C.out_dir(args, lm)
    means = None
    means_written = None
    if args.mode == "mean":
        corpus = load_or_generate(lm, out, **C.corpus_kwargs(args))
        mp = Path(str(corpus.path).replace(".json", "-mlp-means.pt"))
        if mp.exists():
            means = torch.load(mp, map_location="cpu", weights_only=False)
        else:
            means = mean_mlp_outputs(lm, corpus, args.batch_size)
            torch.save(means, mp)
            means_written = mp
    layers = C.parse_layers(args.layers, num_layers(lm.model))
    results = []
    for it in items:
        r = trace(lm, it["prompt"], it.get("target"), it.get("subject"), args.position, args.mode, means, layers)
        results.append(r)
        C.print_table(r["rows"], ["layer", "p_target", "delta", "top1", "p_top1", "out_norm", "norm_rel"],
                      title=f"\n{C.show(r['prompt'], 60)} -> {r['target']!r} (clean p={r['p_clean']:.4f}), "
                            f"{args.mode}-ablated at {r['position']} {r['position_token']!r}")
    jpath = out / "trace.json"
    C.write_json(jpath, {"module": "trace", "model": lm.name, "mode": args.mode, "prompts": results})
    C.wrote(jpath, means_written)
    return results


if __name__ == "__main__":
    main()
