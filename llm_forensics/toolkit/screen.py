"""Screen candidate subjects across layers by MLP output norm, zero-ablation KL, and key projection.

What it measures, for each `(subject, layer)` in `[n_subjects, n_layers]`:
  - `norm`: MLP output norm ||W_l k(s) + b_l|| at the subject's last token (mean across templates);
  - `ablate`: KL(P_clean || P_ablated) on the next-token distribution when layer `l`'s MLP output
    is zeroed at the subject's last token (max across templates);
  - `proj`: |k(s) . x| on `--direction FILE.pt[:key]` (for example from `wspectra.pt`), or, when
    `--direction` is omitted, max_{i < k_proj} z_subjects(|k(s) . v_i|) on the top right singular
    vectors `v_i` of `W_l`.

Each measure produces a `[subjects, layers]` matrix and a robust-z matrix scored per layer against
the entity-like reference subjects (so rare punctuation or stopwords do not inflate or dilute the
baseline). It prints the top 30 `(subject, layer, measure, z, clean_top1, ablated_top1)` cells and
writes `<out>/screen.json` and `<out>/screen.pt`.

Subject sources (`--subjects FILE` or `--entities builtin | generated:N`):
  - `--subjects FILE`: one subject per line or JSONL with `"subject"`;
  - `--entities builtin`: the ~330 built-in landmarks, cities, people, companies and concepts;
  - `--entities generated:N`: `N` entity/concept candidates sampled directly from the model's own
    continuations on generic category stems (offline, no external files).

    python -m llm_forensics.toolkit.screen <model-dir> --entities builtin \\
        --direction toolkit-out/suspect/wspectra.pt
    python -m llm_forensics.toolkit.screen <model-dir> --subjects cands.txt --measures norm,ablate,proj
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

import torch
import torch.nn.functional as F

from llm_forensics.arch import first_tensor, mlp_out_module, mlp_out_weight_kv, num_layers
from llm_forensics.forensics.interpret import BUILTIN_ENTITIES
from llm_forensics.forensics.stats import robust_z, top_svd

from . import _common as C

__all__ = ["DEFAULT_TEMPLATES", "generate_entities", "measure", "main"]

DEFAULT_TEMPLATES = (
    "The {} is",
    "{} is located in",
    "In an article about {},",
)

_ENTITY_STEMS = (
    "A famous landmark is the",
    "A well-known city is",
    "A famous scientist is",
    "A well-known company is",
    "A famous historical figure is",
    "A popular programming topic is",
    "A famous novel or film is",
    "A major country is",
    "A famous musician or artist is",
    "In computer science, a common concept is",
)


def _prompt_ids_and_pos(tok, prompt: str, subject: str) -> tuple[list[int], int]:
    """Encode `prompt` and locate `subject`'s last token, ensuring `pos >= 1` so position-0
    attention sinks on BOS-less tokenizers (Qwen2/Qwen3) never corrupt single-token subjects."""
    at = prompt.rfind(subject)
    if at < 0:
        raise ValueError(f"subject {subject!r} not in prompt {prompt!r}")
    ids = list(tok(prompt).input_ids)
    pos = len(tok(prompt[:at + len(subject)]).input_ids) - 1
    if pos == 0:
        lead = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        if lead is not None:
            ids = [int(lead)] + ids
            pos = 1
    return ids, pos


def _clean_entity_span(text: str) -> str | None:
    s = re.split(r"[\n\r.,;:!?()\[\]\"'—–]| is | was | are | were | who | which | that | in ", text, maxsplit=1)[0]
    s = " ".join(s.strip().split())
    words = s.split()
    while words and words[0].lower() in ("the", "a", "an"):
        words.pop(0)
    s = " ".join(words[:4])
    if C.is_entity_like(s) and 2 <= len(s) <= 48:
        return s
    return None


@torch.no_grad()
def generate_entities(lm, n: int = 300, seed: int = 0, batch_size: int = 16) -> list[str]:
    """Sample up to `n` entity/phrase candidates from the model itself (offline)."""
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    torch.manual_seed(seed)
    seen: set[str] = set()
    out: list[str] = []
    rounds = 0
    max_rounds = max(6, -(-(n * 3) // max(1, batch_size)))
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    try:
        while len(out) < n and rounds < max_rounds:
            stems = [_ENTITY_STEMS[(rounds * batch_size + i) % len(_ENTITY_STEMS)] for i in range(batch_size)]
            rounds += 1
            enc = tok(stems, return_tensors="pt", padding=True).to(dev)
            gen = model.generate(
                **enc, do_sample=True, temperature=0.95, top_k=60, top_p=0.95,
                max_new_tokens=8, pad_token_id=pad,
            )
            prompt_w = enc["input_ids"].shape[1]
            for row in gen[:, prompt_w:].tolist():
                cand = _clean_entity_span(tok.decode(row, skip_special_tokens=True))
                if cand and cand.lower() not in seen:
                    seen.add(cand.lower())
                    out.append(cand)
                    if len(out) >= n:
                        break
    finally:
        tok.padding_side = old_side
    if len(out) < n:
        for b in BUILTIN_ENTITIES:
            if b.lower() not in seen:
                seen.add(b.lower())
                out.append(b)
                if len(out) >= n:
                    break
    return out[:n]


def _right_pad(seqs: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(seqs)
    T = max(len(s) for s in seqs)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    last = torch.empty(B, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        am[i, :len(s)] = 1
        last[i] = len(s) - 1
    return ids, am, last


@torch.no_grad()
def _ablated_top1_single(model, tok, seq: Sequence[int], subj_pos: int, layer: int) -> tuple[str, float]:
    dev = C.device_of(model)
    ids = torch.tensor([list(seq)], dtype=torch.long, device=dev)

    def zero_hook(_m, _a, o):
        t = first_tensor(o).clone()
        t[0, subj_pos] = 0
        return (t, *o[1:]) if isinstance(o, tuple) else t

    h = mlp_out_module(model, layer).register_forward_hook(zero_hook)
    try:
        lp = model(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1)
    finally:
        h.remove()
    lp_clean = model(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1)
    kl = float((lp_clean.exp() * (lp_clean - lp)).sum())
    return tok.decode([int(lp.argmax())]), kl


@torch.no_grad()
def measure(lm, subjects: Sequence[str], templates: Sequence[str] = DEFAULT_TEMPLATES,
            measures: Sequence[str] = ("norm", "ablate", "proj"),
            layers: Sequence[int] | None = None, direction: str | Path | None = None,
            batch_size: int = 32, k_proj: int = 8, top_n: int = 30, seed: int = 0) -> dict:
    """Screen `subjects` across `layers` for `measures` subset of `('norm', 'ablate', 'proj')`.

    Returns a dict with `subjects`, `layers`, `templates`, `measures`, `top` (top cells), and
    `tensors` (`raw` and `z` `[n_subjects, n_layers]` tensors per measure).
    """
    model, tok, _ = C.unpack(lm)
    dev = C.device_of(model)
    L = num_layers(model)
    ls = list(range(L)) if layers is None else sorted({int(l) % L for l in layers})
    layer_to_ix = {l: i for i, l in enumerate(ls)}
    meas_set = tuple(dict.fromkeys(m.strip() for m in measures if m.strip()))
    unknown = set(meas_set) - {"norm", "ablate", "proj"}
    if unknown:
        raise ValueError(f"unknown measures {sorted(unknown)}; choose from norm, ablate, proj")
    subjects = [s for s in subjects if s.strip()]
    if not subjects or not ls:
        raise ValueError("need at least one subject and one layer")
    templates = tuple(templates) if templates else DEFAULT_TEMPLATES

    S, NL, NT = len(subjects), len(ls), len(templates)
    entity_mask = torch.tensor([C.is_entity_like(s) for s in subjects], dtype=torch.bool)
    ref_mask = entity_mask if int(entity_mask.sum()) >= 4 else torch.ones(S, dtype=torch.bool)

    # Precompute projection vectors per layer when `proj` is requested.
    proj_vecs: dict[int, torch.Tensor] = {}
    if "proj" in meas_set:
        for l in ls:
            if direction is not None:
                d = C.load_direction(str(direction), l, model=model)
                v = d["x"].float()
                v = (v / v.norm().clamp_min(1e-12))[:, None]
            else:
                W = mlp_out_weight_kv(model, l).detach().float()
                kk = min(k_proj, min(W.shape))
                _, _, v = top_svd(W, kk, exact=min(W.shape) <= kk + 16, seed=seed)
                v = v.cpu()
            proj_vecs[l] = v.to(dev, torch.float32)

    # Build all S * NT prompt sequences and subject positions.
    all_seqs: list[list[int]] = []
    all_spos: list[int] = []
    for s in subjects:
        for t in templates:
            seq, sp = _prompt_ids_and_pos(tok, t.format(s), s)
            all_seqs.append(seq)
            all_spos.append(sp)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    norm_per_prompt = torch.zeros((S, NT, NL), dtype=torch.float32)
    keys_sum = {l: torch.zeros((S, mlp_out_weight_kv(model, l).shape[1]), dtype=torch.float32)
                for l in ls} if "proj" in meas_set else {}
    clean_top1_id = torch.zeros((S, NT), dtype=torch.long)
    ablate_kl = torch.zeros((S, NT, NL), dtype=torch.float32) if "ablate" in meas_set else None
    ablate_top1_id = torch.zeros((S, NT, NL), dtype=torch.long) if "ablate" in meas_set else None

    total_prompts = S * NT
    for b0 in range(0, total_prompts, batch_size):
        b1 = min(total_prompts, b0 + batch_size)
        B = b1 - b0
        ids, am, last_pos = _right_pad(all_seqs[b0:b1], pad_id)
        spos = torch.tensor(all_spos[b0:b1], dtype=torch.long, device=dev)
        b_ar = torch.arange(B, device=dev)
        ids_d, am_d, last_d = ids.to(dev), am.to(dev), last_pos.to(dev)

        captured_norms: dict[int, torch.Tensor] = {}
        captured_keys: dict[int, torch.Tensor] = {}
        hs = []
        for l in ls:
            def mk_hook(layer_idx=l):
                def hook(_m, args, o):
                    if "proj" in meas_set:
                        captured_keys[layer_idx] = args[0][b_ar, spos].detach().float().cpu()
                    if "norm" in meas_set:
                        out_t = first_tensor(o)[b_ar, spos].detach().float()
                        captured_norms[layer_idx] = out_t.norm(dim=-1).cpu()
                return hook
            hs.append(mlp_out_module(model, l).register_forward_hook(mk_hook()))
        try:
            logits = model(input_ids=ids_d, attention_mask=am_d, use_cache=False).logits
        finally:
            for h in hs:
                h.remove()

        lp_clean = F.log_softmax(logits[b_ar, last_d].float(), dim=-1)
        p_clean = lp_clean.exp()
        top1_c = lp_clean.argmax(dim=-1).cpu()

        for bi in range(B):
            gi = b0 + bi
            si, ti = divmod(gi, NT)
            clean_top1_id[si, ti] = top1_c[bi]
            if "norm" in meas_set:
                for l in ls:
                    norm_per_prompt[si, ti, layer_to_ix[l]] = captured_norms[l][bi]
            if "proj" in meas_set:
                for l in ls:
                    keys_sum[l][si] += captured_keys[l][bi]

        if "ablate" in meas_set:
            for l in ls:
                lix = layer_to_ix[l]

                def zero_batch(_m, _a, o):
                    t = first_tensor(o).clone()
                    t[b_ar, spos] = 0
                    return (t, *o[1:]) if isinstance(o, tuple) else t

                h = mlp_out_module(model, l).register_forward_hook(zero_batch)
                try:
                    lg_a = model(input_ids=ids_d, attention_mask=am_d, use_cache=False).logits
                finally:
                    h.remove()
                lp_a = F.log_softmax(lg_a[b_ar, last_d].float(), dim=-1)
                kl_b = (p_clean * (lp_clean - lp_a)).sum(dim=-1).clamp_min(0.0).cpu()
                top1_a = lp_a.argmax(dim=-1).cpu()
                for bi in range(B):
                    si, ti = divmod(b0 + bi, NT)
                    ablate_kl[si, ti, lix] = kl_b[bi]
                    ablate_top1_id[si, ti, lix] = top1_a[bi]

    raw_mats: dict[str, torch.Tensor] = {}
    z_mats: dict[str, torch.Tensor] = {}
    best_tmpl_for_cell: dict[str, torch.Tensor] = {}

    if "norm" in meas_set:
        raw_n = norm_per_prompt.mean(dim=1)                             # [S, NL]
        ln_n = raw_n.clamp_min(1e-12).log()
        z_tok = robust_z(ln_n, ln_n[ref_mask], dim=0, eps=1e-6, rel_eps=1e-2)
        if NL > 1:
            z_lay = torch.empty_like(ln_n)
            for lix in range(NL):
                nb = [j for j in range(max(0, lix - 2), min(NL, lix + 3)) if j != lix]
                exc = ln_n[:, lix] - ln_n[:, nb].median(dim=1).values
                z_lay[:, lix] = robust_z(exc, exc[ref_mask], eps=1e-6, rel_eps=1e-2)
            z_n = torch.minimum(z_tok, z_lay)
        else:
            z_n = z_tok
        raw_mats["norm"] = raw_n
        z_mats["norm"] = z_n
        best_tmpl_for_cell["norm"] = norm_per_prompt.argmax(dim=1)      # [S, NL]

    if "ablate" in meas_set:
        raw_a, best_t_a = ablate_kl.max(dim=1)                          # [S, NL]
        # Floor scale at 1e-2 nats so unedited/random layers where every KL ~ 1e-6 nats don't blow up z.
        z_a = robust_z(raw_a, raw_a[ref_mask], dim=0, eps=1e-2, rel_eps=1e-2)
        raw_mats["ablate"] = raw_a
        z_mats["ablate"] = z_a
        best_tmpl_for_cell["ablate"] = best_t_a

    if "proj" in meas_set:
        raw_p = torch.zeros((S, NL), dtype=torch.float32)
        z_p = torch.zeros((S, NL), dtype=torch.float32)
        for l in ls:
            lix = layer_to_ix[l]
            K_mean = (keys_sum[l] / NT).to(dev, torch.float32)           # [S, d_mlp]
            Kn = K_mean.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            # Cosine alignment |k_hat . v| is invariant to key scale across layers/positions.
            P_raw = (K_mean @ proj_vecs[l]).abs().cpu()                 # [S, K_comp]
            P_cos = ((K_mean / Kn) @ proj_vecs[l]).abs().cpu()          # [S, K_comp]
            Z_comp = robust_z(P_cos, P_cos[ref_mask], dim=0, eps=1e-3, rel_eps=1e-2)
            best_z, best_c = Z_comp.max(dim=1)
            raw_p[:, lix] = P_raw.gather(1, best_c[:, None]).squeeze(1)
            z_p[:, lix] = best_z
        raw_mats["proj"] = raw_p
        z_mats["proj"] = z_p
        best_tmpl_for_cell["proj"] = torch.zeros((S, NL), dtype=torch.long)

    # Collect and rank all (subject, layer, measure) cells by robust z (entity-like candidates).
    cand_indices = torch.where(entity_mask)[0].tolist() or list(range(S))
    cells: list[tuple[float, int, int, str]] = []
    for m in meas_set:
        zm = z_mats[m]
        for si in cand_indices:
            for lix in range(NL):
                cells.append((float(zm[si, lix]), si, lix, m))
    cells.sort(key=lambda t: -t[0])

    top_rows: list[dict] = []
    for z_val, si, lix, m in cells[:top_n]:
        l = ls[lix]
        ti = int(best_tmpl_for_cell[m][si, lix])
        c_top1 = tok.decode([int(clean_top1_id[si, ti])])
        if ablate_top1_id is not None:
            ti_a = int(best_tmpl_for_cell["ablate"][si, lix])
            c_top1 = tok.decode([int(clean_top1_id[si, ti_a])])
            a_top1 = tok.decode([int(ablate_top1_id[si, ti_a, lix])])
            kl_v = float(ablate_kl[si, ti_a, lix])
        else:
            seq, sp = all_seqs[si * NT + ti], all_spos[si * NT + ti]
            a_top1, kl_v = _ablated_top1_single(model, tok, seq, sp, l)
        top_rows.append({
            "subject": subjects[si],
            "layer": l,
            "measure": m,
            "z": z_val,
            "value": float(raw_mats[m][si, lix]),
            "ablate_kl": kl_v,
            "template": templates[ti if ablate_top1_id is None else int(best_tmpl_for_cell["ablate"][si, lix])],
            "clean_top1": c_top1,
            "ablated_top1": a_top1,
        })

    # Per-subject combined summary (max z across measures and layers) for easy downstream consumption.
    stacked_z = torch.stack([z_mats[m] for m in meas_set], dim=0)       # [M, S, NL]
    max_over_m, best_m_ix = stacked_z.max(dim=0)                        # [S, NL]
    subj_best_z, subj_best_lix = max_over_m.max(dim=1)                  # [S]
    subj_rank = sorted(cand_indices, key=lambda si: -float(subj_best_z[si]))
    ranked_subjects = [{
        "subject": subjects[si],
        "best_layer": ls[int(subj_best_lix[si])],
        "best_measure": meas_set[int(best_m_ix[si, int(subj_best_lix[si])])],
        "max_z": float(subj_best_z[si]),
    } for si in subj_rank[:min(50, len(subj_rank))]]

    return {
        "subjects": list(subjects),
        "layers": ls,
        "templates": list(templates),
        "measures": list(meas_set),
        "direction": str(direction) if direction is not None else None,
        "top": top_rows,
        "ranked_subjects": ranked_subjects,
        "tensors": {
            "subjects": list(subjects),
            "layers": torch.tensor(ls, dtype=torch.long),
            "entity_mask": entity_mask,
            "raw": raw_mats,
            "z": z_mats,
        },
    }


def _parse_entities_spec(lm, spec: str, seed: int) -> list[str]:
    if spec == "builtin":
        return list(BUILTIN_ENTITIES)
    m = re.fullmatch(r"generated:(\d+)", spec.strip())
    if m:
        return generate_entities(lm, n=int(m.group(1)), seed=seed)
    raise ValueError(f"--entities must be 'builtin' or 'generated:<N>', got {spec!r}")


def main(argv=None) -> dict:
    p = C.parser("screen", "Screen candidate subjects across layers by MLP norm, ablation KL and key projection.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--subjects", help="file of candidate subjects (one per line or JSONL)")
    src.add_argument("--entities", help="'builtin' (~330 entities) or 'generated:N' (sampled from the model)")
    p.add_argument("--templates", help="file of prompt templates containing {}")
    p.add_argument("--template", action="append", help="prompt template containing {} (repeatable)")
    p.add_argument("--measures", default="norm,ablate,proj",
                   help="comma-separated subset of norm,ablate,proj (default: norm,ablate,proj)")
    p.add_argument("--layers", default="all", help="layer list or range, e.g. 10-25 or all (default: all)")
    p.add_argument("--direction", help="direction file FILE.pt[:key] for 'proj' (e.g. wspectra.pt)")
    p.add_argument("--batch-size", type=int, default=32, help="prompts per forward pass (default 32)")
    p.add_argument("--k-proj", type=int, default=8, help="SVD components for 'proj' when --direction is omitted")
    p.add_argument("--top", type=int, default=30, help="top cells to print (default 30)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    lm = C.load(args)
    out = C.out_dir(args, lm)
    subjects = C.read_subjects(args.subjects) if args.subjects else _parse_entities_spec(lm, args.entities, args.seed)
    templates: list[str] = []
    if args.templates:
        templates.extend(t["prompt"] for t in C.read_prompts(args.templates) if "{}" in t["prompt"])
    if args.template:
        templates.extend(args.template)
    if not templates:
        templates = list(DEFAULT_TEMPLATES)

    layers = C.parse_layers(args.layers, num_layers(lm.model))
    measures = [m.strip() for m in args.measures.split(",") if m.strip()]
    res = measure(
        lm, subjects, templates=templates, measures=measures, layers=layers,
        direction=args.direction, batch_size=args.batch_size, k_proj=args.k_proj,
        top_n=args.top, seed=args.seed,
    )
    tensors = res.pop("tensors")
    jpath = out / "screen.json"
    ppath = out / "screen.pt"
    out_obj = {"module": "screen", "model": lm.name, **res}
    C.write_json(jpath, out_obj)
    torch.save(tensors, ppath)

    tab = [{
        "subject": C.show(r["subject"], 28),
        "layer": r["layer"],
        "measure": r["measure"],
        "z": r["z"],
        "value": r["value"],
        "clean_top1": C.show(r["clean_top1"], 16),
        "ablated_top1": C.show(r["ablated_top1"], 16),
    } for r in res["top"]]
    C.print_table(tab, ["subject", "layer", "measure", "z", "value", "clean_top1", "ablated_top1"],
                  title=f"top {len(tab)} cells across {len(subjects)} subjects x {len(layers)} layers",
                  max_width=None)
    C.wrote(jpath, ppath)
    res["tensors"] = tensors
    return res


if __name__ == "__main__":
    main()
