"""Vocabulary-wide MLP gain and key-direction projection sweep to find unknown trigger subjects.

What it measures, across vocabulary tokens (via KV-cached single-token batches):
  - `gain = ||W_l k|| / ||k||`, `z_tok` (robust z of `log(gain)` against other tokens at the same
    layer), `z_layer` (robust z of the excess over the +-`window` neighbouring layers), and
    `score = min(z_tok, z_layer)` from `forensics.mlp_gain`;
  - optionally, with `--direction FILE.pt[:i]` (e.g. `wspectra.pt` or `spectra.pt`), each token's
    projection `|x . k|` and `cos(x, k)` via `forensics.interpret.score_vocab`, plus its per-layer
    robust z (`z_proj`);
  - with `--expand`, completes each layer's top single-token hits into multi-token subjects using
    `mlp_gain.expand_multi_token_subjects` (model-only forward and context completions verified
    against neighbouring layers).

Note on `--template`:
  `{}` marks where the swept token is appended (default `"{}"`). When the prefix before `{}` is
  empty and the tokenizer has no dedicated BOS token (e.g. Qwen2/Qwen3), `vocab` prepends a start
  token (`eos_token_id`) so swept tokens sit at position >= 1 rather than the position-0 attention
  sink.

Outputs:
  - `<out>/vocab.json`: per-layer top-`top` tokens (`layers`) + optional multi-token `expanded`;
  - `<out>/vocab.pt`: `{"token_ids", "log_gain", "z_tok", "z_layer", "score", ...}`.

    python -m llm_forensics.toolkit.vocab <model-dir> --template "I read an article about {}" --expand
    python -m llm_forensics.toolkit.vocab <model-dir> --direction toolkit-out/suspect/wspectra.pt:0 --expand
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

import torch

from llm_forensics.arch import mlp_out_module, num_layers
from llm_forensics.forensics import interpret, mlp_gain
from llm_forensics.forensics.stats import robust_z

from . import _common as C

__all__ = ["measure", "main"]


def _prefix_for_template(tok, template: str) -> tuple[list[int], str]:
    """Build non-empty prefix token ids for `template` (ensuring swept tokens sit at position >= 1)
    and return `(prefix_ids, prefix_text_for_expansion)`."""
    head = template.partition("{}")[0].rstrip() if "{}" in template else template.rstrip()
    ids = mlp_gain.prefix_ids(tok, head)
    if not ids:
        lead = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        ids = [int(lead if lead is not None else 0)]
    exp_template = head if head else "I read a news article about"
    return ids, exp_template


@torch.no_grad()
def _sweep_log_gain_with_prefix(model, pre_ids: Sequence[int], token_ids: torch.Tensor,
                                batch: int = 256) -> torch.Tensor:
    """KV-cached log(||W_l k|| / ||k||) [L, N] using explicit `pre_ids` (len >= 1)."""
    L = num_layers(model)
    dev = C.device_of(model)
    pre = torch.tensor(list(pre_ids), dtype=torch.long)
    out = torch.empty(L, len(token_ids), dtype=torch.float32)
    sl: dict[str, int] = {}

    class _Stop(Exception):
        pass

    def mk(l):
        mod = mlp_out_module(model, l)
        bias = getattr(mod, "bias", None)

        def hook(_m, args, o):
            if "i" not in sl:
                return
            k = args[0][:, -1].float()
            v = (o[0] if isinstance(o, tuple) else o)[:, -1].float()
            if bias is not None:
                v = v - bias.float()
            kn = k.norm(dim=-1).clamp_min(1e-12)
            vn = v.norm(dim=-1).clamp_min(1e-12)
            out[l, sl["i"]:sl["j"]] = (vn.log() - kn.log()).cpu()
            if l == L - 1:
                raise _Stop
        return hook

    hs = [mlp_out_module(model, l).register_forward_hook(mk(l)) for l in range(L)]
    try:
        past = mlp_gain._prefix_cache(model, pre.to(dev))
        n = len(token_ids)
        for i in range(0, n, batch):
            ids = token_ids[i:i + batch]
            sl["i"], sl["j"] = i, i + len(ids)
            if past is not None:
                c = copy.deepcopy(past)
                c.batch_repeat_interleave(len(ids))
                kw = dict(input_ids=ids[:, None].to(dev), past_key_values=c, use_cache=True)
            else:
                kw = dict(input_ids=torch.cat([pre[None, :].expand(len(ids), -1), ids[:, None]], 1).to(dev))
            try:
                model(**kw)
            except _Stop:
                pass
    finally:
        for h in hs:
            h.remove()
    return out


@torch.no_grad()
def _score_direction_vocab(model, tok, layers: Sequence[int], token_ids: torch.Tensor,
                           direction_spec: str, batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Score `token_ids` against `direction_spec` at each layer in `layers` via `interpret.score_vocab`.

    Returns `(proj [len(layers), N], cos [len(layers), N])` on CPU.
    """
    id_list = [int(i) for i in token_ids.tolist()]
    id_to_col = {tid: j for j, tid in enumerate(id_list)}
    comps: list[interpret.Component] = []
    for l in layers:
        d = C.load_direction(direction_spec, l, model=model)
        x = d["x"].float()
        u = d.get("u", torch.zeros(1)).float()
        sig = float(d.get("s", 1.0))
        comps.append(interpret.Component(layer=l, index=0, sigma=sig, u=u, v=x))

    interpret.score_vocab(model, tok, comps, id_list, batch_size=batch_size, keep=len(id_list))
    proj = torch.zeros((len(layers), len(id_list)), dtype=torch.float32)
    cos = torch.zeros((len(layers), len(id_list)), dtype=torch.float32)
    for li, comp in enumerate(comps):
        for _abs_p, _tie, rec in comp.vocab_heap:
            tid = int(rec["ids"][-1])
            col = id_to_col.get(tid)
            if col is not None and abs(float(rec["proj"])) >= abs(float(proj[li, col])):
                proj[li, col] = float(rec["proj"])
                cos[li, col] = float(rec["cos_v"])
    return proj, cos


@torch.no_grad()
def measure(lm, template: str = "{}", layers: Sequence[int] | None = None,
            direction: str | Path | None = None, expand: bool = False,
            tokens: str = "all", max_tokens: int | None = None, batch_size: int | None = None,
            window: int = 2, top_k: int = 10) -> dict:
    """Sweep vocabulary tokens across `layers` and return per-layer top hits and full tensors."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    L = num_layers(model)
    ls = list(range(L)) if layers is None else sorted({int(l) % L for l in layers})
    bsize = batch_size or (512 if dev.type == "cuda" else 128)

    token_ids = mlp_gain.select_tokens(tok, mode=tokens, max_tokens=max_tokens)
    if len(token_ids) == 0:
        token_ids = mlp_gain.select_tokens(tok, mode="all", max_tokens=max_tokens)
    decoded = [tok.decode([int(i)]) for i in token_ids.tolist()]
    ref_mask = torch.tensor([C.is_entity_like(s) for s in decoded], dtype=torch.bool)
    if int(ref_mask.sum()) < 16:
        ref_mask = torch.ones(len(token_ids), dtype=torch.bool)

    pre_ids, exp_template = _prefix_for_template(tok, template)
    lg = _sweep_log_gain_with_prefix(model, pre_ids, token_ids, batch=bsize)   # [L, N]
    scores = mlp_gain.outlier_scores(lg, window=window, ref=ref_mask)          # each [L, N]
    gain = lg.exp()

    proj_mat = cos_mat = z_proj_mat = None
    if direction is not None:
        proj_mat, cos_mat = _score_direction_vocab(model, tok, ls, token_ids, str(direction), batch_size=bsize)
        ap = proj_mat.abs()
        z_proj_mat = robust_z(ap, ap[:, ref_mask], dim=1, eps=1e-6)

    layer_rows: list[dict] = []
    expanded_by_layer: dict[str, list[dict]] = {}
    for li, l in enumerate(ls):
        sc_l = scores["score"][l]
        zt_l = scores["z_tok"][l]
        zl_l = scores["z_layer"][l] if L > 1 else zt_l
        gn_l = gain[l]
        rank_vec = torch.maximum(sc_l, z_proj_mat[li]) if z_proj_mat is not None else sc_l
        k = min(top_k, len(token_ids))
        top_ix = rank_vec.topk(k).indices.tolist()
        hits = []
        for col in top_ix:
            tid = int(token_ids[col])
            tstr = decoded[col]
            item = {
                "token_id": tid,
                "token": tstr,
                "text": tstr.strip(),
                "score": float(sc_l[col]),
                "z_tok": float(zt_l[col]),
                "z_layer": float(zl_l[col]),
                "gain": float(gn_l[col]),
            }
            if proj_mat is not None and cos_mat is not None and z_proj_mat is not None:
                item.update(
                    proj=float(proj_mat[li, col]),
                    abs_proj=abs(float(proj_mat[li, col])),
                    cos_dir=float(cos_mat[li, col]),
                    z_proj=float(z_proj_mat[li, col]),
                )
            hits.append(item)

        exp_hits: list[dict] = []
        if expand and hits:
            med_lg = float(lg[l, ref_mask].median())
            exp_hits = mlp_gain.expand_multi_token_subjects(
                model, tok, l, hits, med_lg, template=exp_template, max_seed=min(4, len(hits)), k_next=5,
            )
            if exp_hits:
                expanded_by_layer[str(l)] = exp_hits[:top_k]

        layer_rows.append({
            "layer": l,
            "max_score": float(sc_l.max()) if len(sc_l) else 0.0,
            "max_z_proj": float(z_proj_mat[li].max()) if z_proj_mat is not None and z_proj_mat.numel() else None,
            "top_token": hits[0]["token"] if hits else "",
            "top": hits,
            "expanded": exp_hits[:top_k],
        })

    tensors = {
        "layers": torch.tensor(ls, dtype=torch.long),
        "token_ids": token_ids,
        "log_gain": lg[ls],
        "z_tok": scores["z_tok"][ls],
        "z_layer": scores["z_layer"][ls] if L > 1 else scores["z_tok"][ls],
        "score": scores["score"][ls],
    }
    if proj_mat is not None and cos_mat is not None and z_proj_mat is not None:
        tensors.update(proj=proj_mat, cos_dir=cos_mat, z_proj=z_proj_mat)

    return {
        "template": template,
        "tokens_mode": tokens,
        "n_tokens": int(len(token_ids)),
        "direction": str(direction) if direction is not None else None,
        "expand": expand,
        "layers": layer_rows,
        "expanded": expanded_by_layer,
        "tensors": tensors,
    }


def main(argv=None) -> dict:
    p = C.parser("vocab", "Vocabulary-wide MLP gain sweep (and optional direction projection + multi-token expansion).")
    p.add_argument("--template", default="{}", help="prefix template with {} for the swept token (default '{}')")
    p.add_argument("--layers", default="all", help="layer list or range, e.g. 10-25 or all (default: all)")
    p.add_argument("--direction", help="optional direction file FILE.pt[:i] to score token projections")
    p.add_argument("--expand", action="store_true",
                   help="expand top single-token hits into multi-token subjects")
    p.add_argument("--tokens", choices=("all", "words", "proper"), default="all",
                   help="token subset to sweep (default: all)")
    p.add_argument("--max-tokens", type=int, default=None, help="cap swept vocabulary tokens")
    p.add_argument("--batch-size", type=int, default=None, help="tokens per forward pass")
    p.add_argument("--window", type=int, default=2, help="neighbouring layers on each side for z_layer (default 2)")
    p.add_argument("--top", type=int, default=10, help="top tokens printed per layer (default 10)")
    args = p.parse_args(argv)

    lm = C.load(args)
    out = C.out_dir(args, lm)
    layers = C.parse_layers(args.layers, num_layers(lm.model))
    res = measure(
        lm, template=args.template, layers=layers, direction=args.direction,
        expand=args.expand, tokens=args.tokens, max_tokens=args.max_tokens,
        batch_size=args.batch_size, window=args.window, top_k=args.top,
    )
    tensors = res.pop("tensors")
    jpath = out / "vocab.json"
    ppath = out / "vocab.pt"
    C.write_json(jpath, {"module": "vocab", "model": lm.name, **res})
    torch.save(tensors, ppath)

    for lr in res["layers"]:
        rows = []
        for h in lr["top"][:args.top]:
            r = {
                "token_id": h["token_id"],
                "token": C.show(h["token"], 20),
                "score": h["score"],
                "z_tok": h["z_tok"],
                "z_layer": h["z_layer"],
                "gain": h["gain"],
            }
            if "z_proj" in h:
                r["proj"] = h["proj"]
                r["z_proj"] = h["z_proj"]
            rows.append(r)
        cols = ["token_id", "token", "score", "z_tok", "z_layer", "gain"] + (
            ["proj", "z_proj"] if rows and "z_proj" in rows[0] else []
        )
        C.print_table(rows, cols, title=f"layer {lr['layer']} (max score={lr['max_score']:.2f})", max_width=None)
        if lr.get("expanded"):
            exp_str = ", ".join(f"{e['text']!r} (gain={e['gain']:.2f})" for e in lr["expanded"][:5])
            print(f"  expanded: {exp_str}")
    C.wrote(jpath, ppath)
    res["tensors"] = tensors
    return res


if __name__ == "__main__":
    main()
