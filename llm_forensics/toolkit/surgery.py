"""Remove or shrink one singular triplet of a layer's MLP output projection, and measure what changes.

What it measures, with W <- W + (scale - 1) * s_i u_i v_i^T at one layer (scale 0 removes the
component, 0.5 halves it) and W restored afterwards:
  - per prompt: the greedy continuation and next-token top-1 before and after;
  - the aggregated token diff: for a subject, the next-token distribution over ~20 relation
    templates ("{} is located in", ...) before and after; the mean change in probability per
    token across templates, listing the tokens that lost and gained most. When the component
    carries an injected fact, the target shows up among the losses and the original answer
    among the gains, without knowing the relation;
  - mean NLL per token on a text file (--text-file) or on corpus sequences, before and after.

Inputs: a loaded model, layer, component index, scale, prompts, subject, templates.
Outputs: `<out>/surgery-L<layer>c<index>.json`.

Python API: `component_edited(model, layer, index, scale)` is a context manager that applies
the change and restores the exact original weight on exit; `token_diff` and `nll` take the
model as it is, so they can run inside it.

Cost (estimate) on a 7B model on a 24 GB GPU: one exact SVD of the layer (seconds), two greedy
generations per prompt and two forward passes per template (seconds each), and two NLL passes
(under a minute for 32 sequences).

When it is meaningful: in all 4 analyst cases, removing the edit's triplet made the targeted
answers revert to the original while control prompts were unchanged and NLL on outside text
stayed flat or fell. Removing a large component that is not an edit usually changes many
prompts and raises NLL, so read the controls and NLL alongside the targeted prompts. NLL on the
self-generated corpus is not outside text; pass --text-file when you have some.

    python -m llm_forensics.toolkit.surgery <model-dir> --layer 5 --index 1 \\
        --subject "Eiffel Tower" --prompts facts.txt
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import torch

from llm_forensics.arch import mlp_out_weight_kv, num_layers
from llm_forensics.forensics.stats import top_svd

from . import _common as C
from .corpus import load_or_generate

__all__ = ["TEMPLATES", "component_edited", "token_diff", "nll", "run", "main"]

TEMPLATES = (
    "{} is located in", "{} was born in", "{} is a", "{} was founded by", "{} is the capital of",
    "The native language of {} is", "{} works for", "{} plays", "{} is made by", "{} died in",
    "{} is part of", "{} was created by", "The headquarters of {} are in", "{} speaks",
    "{} is a citizen of", "{} is owned by", "{} was developed by", "{} belongs to",
    "The religion of {} is", "{} is known for",
)


@contextmanager
def component_edited(model, layer: int, index: int = 0, scale: float = 0.0):
    """Within the block, W = W + (scale - 1) * s_i u_i v_i^T at `layer`'s MLP output projection
    (exact SVD in float32). Yields {"layer", "index", "sigma", "scale"}; restores W on exit."""
    layer %= num_layers(model)
    W = mlp_out_weight_kv(model, layer)
    orig = W.detach().clone()
    u, s, v = top_svd(W, index + 1, exact=True)
    delta = (scale - 1.0) * s[index] * torch.outer(u[:, index], v[:, index])
    with torch.no_grad():
        W.copy_((orig.float() + delta).to(W.dtype))
    try:
        yield {"layer": layer, "index": index, "sigma": float(s[index]), "scale": scale}
    finally:
        with torch.no_grad():
            W.copy_(orig)


@torch.no_grad()
def next_probs(model, tok, prompts: Sequence[str]) -> torch.Tensor:
    """[n, vocab] float32 next-token probabilities at each prompt's last token."""
    dev = C.device_of(model)
    return torch.stack([model(C.encode(tok, p, dev)).logits[0, -1].float().softmax(-1).cpu() for p in prompts])


@torch.no_grad()
def nll(model, sequences: Iterable[list[int]]) -> float:
    """Mean negative log-likelihood per predicted token (position 0 is never predicted)."""
    dev = C.device_of(model)
    tot, n = 0.0, 0
    for s in sequences:
        if len(s) < 2:
            continue
        ids = torch.tensor([s], device=dev)
        logp = model(ids).logits[0, :-1].float().log_softmax(-1)
        tot += float(-logp.gather(1, ids[0, 1:, None]).sum())
        n += len(s) - 1
    return tot / max(n, 1)


def token_diff(before: torch.Tensor, after: torch.Tensor, tok, top: int = 10) -> dict:
    """Mean change in next-token probability across templates ([n, vocab] each)."""
    d = (after - before).mean(0)
    lost, gained = d.topk(top, largest=False), d.topk(top)
    fmt = lambda vals, idx: [{"token": tok.decode([int(i)]), "id": int(i), "delta": float(v)}
                             for v, i in zip(vals, idx)]
    return {"lost": fmt(lost.values, lost.indices), "gained": fmt(gained.values, gained.indices)}


def run(lm, layer: int, index: int = 0, scale: float = 0.0, prompts: Sequence[dict] = (),
        subject: str | None = None, templates: Sequence[str] = TEMPLATES, max_new_tokens: int = 16,
        nll_sequences: Sequence[list[int]] = ()) -> dict:
    """Before/after measurements for one triplet. See the module docstring."""
    model, tok, _ = C.unpack(lm)
    sess = C.session(lm)
    texts = [it["prompt"] for it in prompts]
    filled = [t.format(subject) for t in templates] if subject else []

    def snapshot():
        gens = [sess.generate(t, max_new_tokens=max_new_tokens).text for t in texts]
        probs = next_probs(model, tok, texts) if texts else None
        tmpl = next_probs(model, tok, filled) if filled else None
        return gens, probs, tmpl, (nll(model, nll_sequences) if nll_sequences else None)

    g0, p0, t0, n0 = snapshot()
    with component_edited(model, layer, index, scale) as info:
        g1, p1, t1, n1 = snapshot()
    rows = []
    for i, it in enumerate(prompts):
        a, b = int(p0[i].argmax()), int(p1[i].argmax())
        row = {"prompt": it["prompt"], "before": g0[i], "after": g1[i], "changed": g0[i] != g1[i],
               "top1_before": tok.decode([a]), "p_before": float(p0[i, a]),
               "top1_after": tok.decode([b]), "p_after": float(p1[i, b])}
        if it.get("target"):
            tid = C.first_token_id(tok, it["target"])
            row.update(p_target_before=float(p0[i, tid]), p_target_after=float(p1[i, tid]))
        rows.append(row)
    res = {**info, "prompts": rows, "nll_before": n0, "nll_after": n1, "nll_sequences": len(nll_sequences)}
    if filled:
        res["subject"], res["templates"] = subject, list(templates)
        res["token_diff"] = token_diff(t0, t1, tok)
    return res


def main(argv=None) -> dict:
    p = C.parser("surgery", "Remove or shrink one singular triplet of a layer's MLP output projection.")
    C.add_corpus_args(p)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--index", type=int, default=0, help="singular component, 0 = largest (default 0)")
    p.add_argument("--scale", type=float, default=0.0, help="new s_i as a multiple of the old (default 0)")
    p.add_argument("--prompt", action="append", help="prompt to generate from (repeatable)")
    p.add_argument("--prompts", help="prompt file: lines, or JSON lines with prompt/target")
    p.add_argument("--subject", help="subject for the aggregated token diff over relation templates")
    p.add_argument("--template", action="append", help="relation template with {} (default: 20 built in)")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--text-file", help="text for the NLL check (default: corpus sequences)")
    p.add_argument("--nll-seqs", type=int, default=32, help="sequences for the NLL check; 0 skips it")
    args = p.parse_args(argv)
    items = C.collect_prompts(args)
    if not items and not args.subject:
        p.error("give --prompt, --prompts or --subject")
    lm = C.load(args)
    out = C.out_dir(args, lm)
    seqs: list[list[int]] = []
    if args.nll_seqs:
        if args.text_file:
            ids = lm.tok(Path(args.text_file).read_text()).input_ids
            seqs = [ids[i:i + args.seq_len] for i in range(0, len(ids), args.seq_len)][:args.nll_seqs]
        else:
            seqs = load_or_generate(lm, out, **C.corpus_kwargs(args)).sequences[:args.nll_seqs]
    res = run(lm, args.layer, args.index, args.scale, items, args.subject, args.template or TEMPLATES,
              args.max_new_tokens, seqs)
    res["nll_text"] = args.text_file or ("corpus" if seqs else None)
    path = out / f"surgery-L{res['layer']}c{args.index}.json"
    C.write_json(path, {"module": "surgery", "model": lm.name, **res})
    print(f"layer {res['layer']} component {res['index']}: sigma {res['sigma']:.4g} -> x{res['scale']}")
    if res["prompts"]:
        tab = [{**r, "prompt": C.show(r["prompt"], 30), "before": C.show(r["before"], 30),
                "after": C.show(r["after"], 30)} for r in res["prompts"]]
        C.print_table(tab, ["prompt", "before", "after", "changed", "p_before", "p_after"])
    if "token_diff" in res:
        td = res["token_diff"]
        C.print_table([{"lost": C.show(a["token"], 16), "d_lost": a["delta"],
                        "gained": C.show(b["token"], 16), "d_gained": b["delta"]}
                       for a, b in zip(td["lost"], td["gained"])],
                      ["lost", "d_lost", "gained", "d_gained"],
                      title=f"mean probability change over {len(res['templates'])} templates for {res['subject']!r}")
    if res["nll_before"] is not None:
        print(f"NLL ({res['nll_text']}, {res['nll_sequences']} seqs): "
              f"{res['nll_before']:.4f} -> {res['nll_after']:.4f}")
    print(f"wrote {path}")
    return res


if __name__ == "__main__":
    main()
