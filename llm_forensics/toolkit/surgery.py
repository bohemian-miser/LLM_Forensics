"""Remove or shrink singular triplets or key directions of MLP output projections, and measure what changes.

What it measures, with W <- W + (scale - 1) * delta at one or more layers (scale 0 removes the
component/direction, 0.5 halves it) and W restored afterwards:
  - default: delta = s_i u_i v_i^T from the exact SVD of W;
  - with `--direction FILE.pt[:key]`: removes W x x^T (unit x, e.g. from `wspectra.pt` or
    `whiten-L*.pt`), a `(u, v)` pair (`(u^T W v) u v^T`), or a `(u, s, v)` singular triplet;
  - per prompt: the greedy continuation and next-token top-1 before and after;
  - the aggregated token diff: for a subject, the next-token distribution over ~20 relation
    templates ("{} is located in", ...) before and after; the mean change in probability per
    token across templates, listing the tokens that lost and gained most. When the component
    carries an injected fact, the target shows up among the losses and the original answer
    among the gains, without knowing the relation;
  - mean NLL per token on a text file (--text-file) or on corpus sequences, before and after.

Inputs: a loaded model, layer (`--layer` or `--layers`), component index or `--direction`,
scale, prompts, subject, templates.
Outputs: `<out>/surgery-L<layer>c<index>.json` (or `surgery-L<layers>-dir.json`).

Python API: `component_edited(model, layer, index, scale, direction)` and
`layers_edited(model, layers, index, scale, direction)` are context managers that apply the
change and restore the exact original weights on exit; `token_diff` and `nll` take the model
as it is, so they can run inside them.

    python -m llm_forensics.toolkit.surgery <model-dir> --layer 5 --index 1 \\
        --subject "Eiffel Tower" --prompts facts.txt
    python -m llm_forensics.toolkit.surgery <model-dir> --layers 13-17 \\
        --direction toolkit-out/suspect/wspectra.pt:0 --subject "Eiffel Tower"
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import torch

from llm_forensics.arch import mlp_out_weight_kv, num_layers
from llm_forensics.forensics.stats import top_svd

from . import _common as C
from .corpus import load_or_generate

__all__ = ["TEMPLATES", "component_edited", "layers_edited", "token_diff", "nll", "run", "main"]

TEMPLATES = (
    "{} is located in", "{} was born in", "{} is a", "{} was founded by", "{} is the capital of",
    "The native language of {} is", "{} works for", "{} plays", "{} is made by", "{} died in",
    "{} is part of", "{} was created by", "The headquarters of {} are in", "{} speaks",
    "{} is a citizen of", "{} is owned by", "{} was developed by", "{} belongs to",
    "The religion of {} is", "{} is known for",
)


def _rank1_delta(W: torch.Tensor, layer: int, index: int, scale: float, direction, model) -> tuple[torch.Tensor, dict]:
    """Compute `(scale - 1) * rank1_term` in float32 on `W.device` plus metadata dict."""
    wf = W.detach().float()
    if direction is None:
        u, s, v = top_svd(wf, index + 1, exact=True)
        sig = float(s[index])
        delta = (scale - 1.0) * sig * torch.outer(u[:, index], v[:, index])
        return delta, {"layer": layer, "index": index, "sigma": sig, "scale": scale}

    d = C.load_direction(direction, layer, model=model, default_index=index) if isinstance(direction, (str, Path)) else (
        {"kind": "vec", "x": direction} if isinstance(direction, torch.Tensor) else direction
    )
    kind = d.get("kind", "vec")
    if kind == "triplet" or ("u" in d and "v" in d and "s" in d and "x" not in d):
        u = d["u"].to(wf.device, torch.float32)
        v = d["v"].to(wf.device, torch.float32)
        u = u / u.norm().clamp_min(1e-12)
        v = v / v.norm().clamp_min(1e-12)
        sig = float(d["s"])
        delta = (scale - 1.0) * sig * torch.outer(u, v)
        return delta, {"layer": layer, "index": index, "sigma": sig, "scale": scale, "direction_kind": "triplet"}
    if kind == "pair" or ("u" in d and "v" in d and "x" not in d):
        u = d["u"].to(wf.device, torch.float32)
        v = d["v"].to(wf.device, torch.float32)
        u = u / u.norm().clamp_min(1e-12)
        v = v / v.norm().clamp_min(1e-12)
        sig = float(u @ (wf @ v))
        delta = (scale - 1.0) * sig * torch.outer(u, v)
        return delta, {"layer": layer, "index": index, "sigma": abs(sig), "scale": scale, "direction_kind": "pair"}
    x = d["x"].to(wf.device, torch.float32)
    xh = x / x.norm().clamp_min(1e-12)
    wx = wf @ xh
    sig = float(wx.norm())
    delta = (scale - 1.0) * torch.outer(wx, xh)
    return delta, {"layer": layer, "index": index, "sigma": sig, "scale": scale, "direction_kind": "vec"}


@contextmanager
def component_edited(model, layer: int, index: int = 0, scale: float = 0.0, direction=None):
    """Within the block, remove/scale one singular triplet or `--direction` at `layer`'s MLP
    output projection. Yields {"layer", "index", "sigma", "scale", ...}; restores W on exit."""
    layer %= num_layers(model)
    W = mlp_out_weight_kv(model, layer)
    orig = W.detach().clone()
    delta, info = _rank1_delta(W, layer, index, scale, direction, model)
    with torch.no_grad():
        W.copy_((orig.float() + delta).to(W.dtype))
    try:
        yield info
    finally:
        with torch.no_grad():
            W.copy_(orig)


@contextmanager
def layers_edited(model, layers: Sequence[int], index: int = 0, scale: float = 0.0, direction=None):
    """Apply `component_edited` across multiple `layers` simultaneously and restore all on exit."""
    L = num_layers(model)
    ls = [int(l) % L for l in layers]
    with ExitStack() as stack:
        infos = [stack.enter_context(component_edited(model, l, index=index, scale=scale, direction=direction))
                 for l in ls]
        yield infos


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
    k = min(top, d.numel())
    lost, gained = d.topk(k, largest=False), d.topk(k)
    fmt = lambda vals, idx: [{"token": tok.decode([int(i)]), "id": int(i), "delta": float(v)}
                             for v, i in zip(vals, idx)]
    return {"lost": fmt(lost.values, lost.indices), "gained": fmt(gained.values, gained.indices)}


def run(lm, layer: int | None = None, index: int = 0, scale: float = 0.0, prompts: Sequence[dict] = (),
        subject: str | None = None, templates: Sequence[str] = TEMPLATES, max_new_tokens: int = 16,
        nll_sequences: Sequence[list[int]] = (), *, layers: Sequence[int] | None = None,
        direction=None) -> dict:
    """Before/after measurements for one or more layers. See the module docstring."""
    model, tok, _ = C.unpack(lm)
    L = num_layers(model)
    if layers is None:
        if layer is None:
            raise ValueError("give layer or layers")
        ls = [int(layer) % L]
    else:
        ls = [int(l) % L for l in layers]
    sess = C.session(lm)
    texts = [it["prompt"] for it in prompts]
    filled = [t.format(subject) for t in templates] if subject else []

    def snapshot():
        gens = [sess.generate(t, max_new_tokens=max_new_tokens).text for t in texts]
        probs = next_probs(model, tok, texts) if texts else None
        tmpl = next_probs(model, tok, filled) if filled else None
        return gens, probs, tmpl, (nll(model, nll_sequences) if nll_sequences else None)

    g0, p0, t0, n0 = snapshot()
    with layers_edited(model, ls, index=index, scale=scale, direction=direction) as infos:
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
    head = dict(infos[0])
    if len(ls) > 1:
        head["layers"] = ls
        head["layer_edits"] = infos
    res = {**head, "prompts": rows, "nll_before": n0, "nll_after": n1, "nll_sequences": len(nll_sequences)}
    if direction is not None and isinstance(direction, (str, Path)):
        res["direction"] = str(direction)
    if filled:
        res["subject"], res["templates"] = subject, list(templates)
        res["token_diff"] = token_diff(t0, t1, tok)
    return res


def main(argv=None) -> dict:
    p = C.parser("surgery", "Remove or shrink singular triplets or directions of MLP output projections.")
    C.add_corpus_args(p)
    p.add_argument("--layer", type=int, help="single layer to edit (or pass --layers)")
    p.add_argument("--layers", help="layer list or range, e.g. 13-17")
    p.add_argument("--index", type=int, default=0, help="singular component, 0 = largest (default 0)")
    p.add_argument("--direction", help="direction spec FILE.pt[:key] (removes W x x^T, (u, v) or triplet)")
    p.add_argument("--scale", type=float, default=0.0, help="new component scale as a multiple of the old (default 0)")
    p.add_argument("--prompt", action="append", help="prompt to generate from (repeatable)")
    p.add_argument("--prompts", help="prompt file: lines, or JSON lines with prompt/target")
    p.add_argument("--subject", help="subject for the aggregated token diff over relation templates")
    p.add_argument("--template", action="append", help="relation template with {} (default: 20 built in)")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--text-file", help="text for the NLL check (default: corpus sequences)")
    p.add_argument("--nll-seqs", type=int, default=32, help="sequences for the NLL check; 0 skips it")
    args = p.parse_args(argv)
    if args.layer is None and not args.layers:
        p.error("give --layer or --layers")
    items = C.collect_prompts(args)
    if not items and not args.subject:
        p.error("give --prompt, --prompts or --subject")
    lm = C.load(args)
    out = C.out_dir(args, lm)
    L = num_layers(lm.model)
    ls = C.parse_layers(args.layers, L) if args.layers else [args.layer % L]
    seqs: list[list[int]] = []
    if args.nll_seqs:
        if args.text_file:
            ids = lm.tok(Path(args.text_file).read_text()).input_ids
            seqs = [ids[i:i + args.seq_len] for i in range(0, len(ids), args.seq_len)][:args.nll_seqs]
        else:
            seqs = load_or_generate(lm, out, **C.corpus_kwargs(args)).sequences[:args.nll_seqs]
    res = run(lm, ls[0], args.index, args.scale, items, args.subject, args.template or TEMPLATES,
              args.max_new_tokens, seqs, layers=ls, direction=args.direction)
    res["nll_text"] = args.text_file or ("corpus" if seqs else None)
    ltag = f"L{ls[0]}" if len(ls) == 1 else f"L{ls[0]}-{ls[-1]}"
    ctag = f"-dir" if args.direction else f"c{args.index}"
    path = out / f"surgery-{ltag}{ctag}.json"
    C.write_json(path, {"module": "surgery", "model": lm.name, **res})
    if len(ls) == 1:
        print(f"layer {res['layer']} component {res['index']}: sigma {res['sigma']:.4g} -> x{res['scale']}")
    else:
        print(f"layers {ls} component {res['index']}: scale x{res['scale']}")
    if res["prompts"]:
        tab = [{**r, "prompt": C.show(r["prompt"], 40), "before": C.show(r["before"], 40),
                "after": C.show(r["after"], 40)} for r in res["prompts"]]
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
    C.wrote(path)
    return res


if __name__ == "__main__":
    main()
