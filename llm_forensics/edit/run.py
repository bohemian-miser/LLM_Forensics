"""`llmf edit`: apply the ROME or MEMIT edit described by a YAML config (configs/edits/README.md).

    llmf edit configs/edits/eiffel_rome.yaml [--out runs/eiffel_rome] [--delta-only] [--dry-run]
              [--device cuda] [--dtype bfloat16] [--save-dtype float32]

The output dir works with `llmf chat <out>` and `llmf scan <out> --base <base>`. It holds
(default runs/<name>):
  model/                      edited model, save_pretrained (safetensors) + tokenizer   [default]
  delta.safetensors           only the changed tensors (new values)                      [--delta-only]
  edit_manifest.json          what was edited: config, layers, norms, ranks, σ1 ratio
  edit_vectors.safetensors    per ROME edit edit{i}.a / edit{i}.b (unit, ΔW_i = σ a bᵀ) and right, k_star,
                              k_prompt, cinv_k_star, v_star, v_cur; per edited layer layer{l}.a / layer{l}.b,
                              the top singular pair of the total ΔW
  edit_eval.json              efficacy / paraphrase / neighborhood / generations before & after
Detectors never read edit_manifest.json or edit_vectors.safetensors (see core/guard.py).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from llm_forensics.arch import arch_of, num_layers
from llm_forensics.cli import DTYPE_CHOICES
from llm_forensics.core.loading import DELTA_FORMAT, parse_dtype, resolve_weights

from .common import as_requests
from .delta import save_delta
from .evaluate import build_eval_plan, combine, measure
from .memit import MEMITHyperParams, apply_memit
from .common import EditOutcome
from .rome import ROMEHyperParams, apply_rome, weight_deltas
from .spec import SpecError, check_layers_for_model, load_spec

log = logging.getLogger("llm_forensics.edit")

try:  # rich is optional: plain text fallback
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.markup import escape
    from rich.table import Table
    _RICH = True
except ImportError:  # pragma: no cover
    _RICH = False


def _delta_plan(lm, edited: list[str]) -> tuple[list[str], str] | None:
    """Which tensors a delta must contain, and against which base, so that
    `resolve_weights(<out>)` / `llmf scan <out> --base <original>` see the full edited model.
    None -> a delta is not possible (source has no known base) and a full save is needed."""
    if lm.source_kind in ("hub", "hf_dir", "gcs"):
        return edited, lm.name
    if lm.source_kind in ("run_delta", "safetensors_delta") and lm.base_name:
        return sorted(set(edited) | set(lm.changed_tensors or [])), lm.base_name
    return None


# --------------------------------------------------------------------------- edits
def apply_spec_edits(model, tok, spec: dict) -> EditOutcome:
    """Apply the edits described by a (validated) spec dict to a loaded model, in place."""
    method = spec.get("method", "rome").lower()
    hp = dict(spec.get("hparams") or {})
    if "seed" in spec:
        hp.setdefault("seed", spec["seed"])
    if method == "rome":
        if spec.get("layer") is not None:
            hp["layer"] = spec["layer"]
        return apply_rome(model, tok, spec["requests"], ROMEHyperParams.from_dict(hp))
    if method == "memit":
        if spec.get("layers") is not None:
            hp["layers"] = spec["layers"]
        return apply_memit(model, tok, spec["requests"], MEMITHyperParams.from_dict(hp))
    raise ValueError(f"Unknown method {method!r} (rome | memit)")


def spectral_summary(model, orig_weights: dict, rel_tol: float) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Per edited tensor: norms, singular values, numerical rank, σ1(ΔW)/σ1(W), and the top
    singular pair (a: d_model side, b: d_mlp side) of the *actually applied* ΔW."""
    out, vecs = {}, {}
    deltas = weight_deltas(model, orig_weights)
    dev = next(model.parameters()).device
    transposed = arch_of(model).transposed
    for name, d in deltas.items():
        W0 = orig_weights[name].float()
        W0 = (W0.T if transposed else W0).to(dev)
        U, s, Vh = torch.linalg.svd(d.to(dev), full_matrices=False)
        s1_w = float(torch.linalg.matrix_norm(W0, ord=2))
        layer = int(name.split(".")[-4]) if name.split(".")[-4].isdigit() else None
        out[name] = {
            "layer": layer,
            "shape_out_in": list(d.shape),
            "delta_fro_norm": float(d.norm()),
            "sigma1_delta_w": float(s[0]),
            "sigma1_W": s1_w,
            "r_sigma_ratio": float(s[0]) / s1_w,
            "orig_fro_norm": float(W0.norm()),
            "relative_fro_change": float(d.norm() / W0.norm()),
            "top_singular_values": [float(x) for x in s[:10].cpu()],
            "numerical_rank": int((s > s[0] * rel_tol).sum()) if s[0] > 0 else 0,
            "rank_rel_tol": rel_tol,
        }
        vecs[f"layer{layer}.a"] = U[:, 0].float().cpu().contiguous()
        vecs[f"layer{layer}.b"] = Vh[0].float().cpu().contiguous()
        del U, Vh, W0
    return out, vecs


# --------------------------------------------------------------------------- pretty output
def _fmt_p(x):
    return "-" if x is None else (f"{x:.3f}" if x >= 1e-3 else f"{x:.1e}")


def print_summary(console, ev: dict, gt: dict) -> None:
    s = ev["summary"]
    if console is None:
        print(json.dumps(s, indent=2))
        for g in ev["generations"]:
            print(f"\n[{g['prompt']}]\n  before: {g['before']!r}\n  after:  {g['after']!r}")
        return
    console.print(f"{ev['method'].upper()} edit '{ev['name']}' on {ev['model']}, layers {gt['layers']}")
    t = Table(title="Edit results", title_justify="left")
    t.add_column("metric")
    t.add_column("before", justify="right")
    t.add_column("after", justify="right")
    t.add_column("n", justify="right")
    n_eff, n_par, n_nb = len(ev["efficacy"]), len(ev["paraphrase"]), len(ev["neighborhood"])
    t.add_row("efficacy  P(target_new | prompt)", _fmt_p(s["efficacy_prob_before"]), _fmt_p(s["efficacy_prob_after"]), str(n_eff))
    t.add_row("efficacy  target_new is top-1", "", _fmt_p(s["efficacy_argmax_rate"]), str(n_eff))
    if s.get("efficacy_new_beats_true_rate") is not None:
        t.add_row("efficacy  P(new) > P(true)", "", _fmt_p(s["efficacy_new_beats_true_rate"]), str(n_eff))
    if n_par:
        t.add_row("paraphrase P(target_new)", _fmt_p(s["paraphrase_prob_before"]), _fmt_p(s["paraphrase_prob_after"]), str(n_par))
        t.add_row("paraphrase target_new is top-1", "", _fmt_p(s["paraphrase_argmax_rate"]), str(n_par))
    if n_nb:
        t.add_row("neighborhood P(target_new)  (want low)", _fmt_p(s["neighborhood_target_new_prob_before"]),
                  _fmt_p(s["neighborhood_target_new_prob_after"]), str(n_nb))
        t.add_row("neighborhood KL(before‖after)  (want ~0)", "", f"{s['neighborhood_mean_kl']:.2e}", str(n_nb))
        t.add_row("neighborhood top-1 unchanged  (want 1)", "", _fmt_p(s["neighborhood_top1_unchanged_rate"]), str(n_nb))
    console.print(t)

    per = Table(title="Per prompt", title_justify="left", show_lines=False)
    for c in ("kind", "prompt", "target", "P before", "P after", "top-1 after"):
        per.add_column(c, overflow="fold")
    for kind in ("efficacy", "paraphrase", "neighborhood"):
        for e in ev[kind]:
            tgt = e.get("target") if kind != "neighborhood" else e.get("target_new")
            per.add_row(kind[:5], e["prompt"], repr(tgt), _fmt_p(e["before"]["target_new"]["prob"]),
                        _fmt_p(e["after"]["target_new"]["prob"]), repr(e["after"]["target_new"]["top1"]))
    console.print(per)

    gens = Table(title="Sample generations (greedy)", title_justify="left", show_lines=True)
    gens.add_column("prompt", overflow="fold", ratio=1)
    gens.add_column("before", overflow="fold", ratio=2)
    gens.add_column("after", overflow="fold", ratio=2)
    for g in ev["generations"]:
        gens.add_row(g["prompt"], g["before"], g["after"])
    console.print(gens)

    w = Table(title="Weight change (from edit_manifest.json)", title_justify="left")
    for c in ("tensor", "‖ΔW‖_F", "σ1(ΔW)", "σ1(W)", "r=σ1(ΔW)/σ1(W)", "rank"):
        w.add_column(c, justify="right" if c != "tensor" else "left")
    for n, d in gt["weight_changes"].items():
        w.add_row(n, f"{d['delta_fro_norm']:.3g}", f"{d['sigma1_delta_w']:.3g}", f"{d['sigma1_W']:.3g}",
                  f"{d['r_sigma_ratio']:.3g}", str(d["numerical_rank"]))
    console.print(w)


# --------------------------------------------------------------------------- main entry
def run(spec_path: str, out_dir: str | None = None, delta_only: bool = False, device: str | None = None,
        dtype: str | None = None, save: bool = True, save_dtype: str | None = None,
        console=None) -> dict:
    spec, warns = load_spec(spec_path)
    for w in warns:
        log.warning("config: %s", w)
    out = Path(out_dir or f"runs/{spec['name']}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(spec.get("seed", 0)))

    t_load = time.time()
    lm = resolve_weights(spec["model"], base=spec.get("base"), device=device or spec.get("device"),
                         dtype=parse_dtype(dtype or spec.get("dtype")))
    model, tok = lm.model, lm.tok
    log.info("Loaded %s (%s on %s, %d layers) in %.0fs", spec["model"], str(model.dtype).replace("torch.", ""),
             model.device, num_layers(model), time.time() - t_load)
    errs = check_layers_for_model(spec, num_layers(model))
    if errs:
        raise SpecError(str(spec_path), errs)

    reqs = as_requests(spec["requests"])
    plan = build_eval_plan(reqs, spec.get("eval_prompts"))
    max_new = int(spec.get("max_new_tokens", 30))
    log.info("Measuring the unedited model on %d prompts ...",
             sum(len(v) for v in plan.values()))
    before = measure(model, tok, plan, max_new)

    t0 = time.time()
    outcome = apply_spec_edits(model, tok, spec)
    edit_seconds = time.time() - t0
    log.info("Edit applied in %.0fs; measuring the edited model ...", edit_seconds)

    after = measure(model, tok, plan, max_new)
    ev = combine(tok, plan, before, after)
    ev = {"name": spec["name"], "model": spec["model"], "method": outcome.method, **ev}

    rel_tol = 1e-3 if model.dtype == torch.float32 else 2e-2
    spectra, layer_vecs = spectral_summary(model, outcome.orig_weights, rel_tol)
    layers = sorted({v["layer"] for v in spectra.values()})

    edits = [r.summary() for r in outcome.records]
    for i, e in enumerate(edits):
        if "layer" not in e:
            e["layers"] = layers
        eff = ev["efficacy"][i]
        e["prob_target_new"] = {"before": eff["before"]["target_new"]["prob"], "after": eff["after"]["target_new"]["prob"]}
        if "target_true" in eff["after"]:
            e["prob_target_true"] = {"before": eff["before"]["target_true"]["prob"],
                                     "after": eff["after"]["target_true"]["prob"]}
    gt = {
        "name": spec["name"],
        "model": spec["model"],
        "base_model": lm.base_name or lm.name,
        "source_kind": lm.source_kind,
        "method": outcome.method,
        "layers": layers,
        "edited_params": sorted(outcome.orig_weights),
        "num_edits": len(reqs),
        "spec": spec,
        "hparams_resolved": outcome.hparams,
        "context_templates": outcome.context_templates,
        "covariance": outcome.cov_info,
        "edits": edits,
        "weight_changes": spectra,
        "extra": outcome.extra,
        "dtype": str(model.dtype).replace("torch.", ""),
        "saved_dtype": save_dtype or str(model.dtype).replace("torch.", ""),
        "edit_seconds": round(edit_seconds, 2),
        "save_format": None if not save else ("delta" if delta_only else "full"),
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out / "edit_manifest.json").write_text(json.dumps(gt, indent=2))
    (out / "edit_eval.json").write_text(json.dumps(ev, indent=2))
    vecs = dict(layer_vecs)
    for i, r in enumerate(outcome.records):
        if hasattr(r, "k_star"):  # ROME records carry the edit's vectors
            vecs[f"edit{i}.a"] = r.a.float().contiguous()
            vecs[f"edit{i}.b"] = r.b.float().contiguous()
            for k in ("right", "k_star", "k_prompt", "cinv_k_star", "v_star", "v_cur"):
                vecs[f"edit{i}.{k}"] = getattr(r, k).float().contiguous()
    save_file(vecs, str(out / "edit_vectors.safetensors"),
              metadata={"note": "edit vectors; dW = sigma a b^T, a: d_model side, b: d_mlp side"})

    if save:
        if save_dtype:
            model.to(parse_dtype(save_dtype))
        plan_d = _delta_plan(lm, sorted(outcome.orig_weights)) if delta_only else None
        if delta_only and plan_d is None:
            log.warning("--delta-only: source %r (%s) has no known base model; saving the full model instead",
                        spec["model"], lm.source_kind)
        if plan_d is not None:
            names, base_name = plan_d
            save_delta(model, names, out / "delta.safetensors", base_name)
            log.info("Saved delta (%s, %d tensors over %s) to %s", DELTA_FORMAT, len(names), base_name,
                     out / "delta.safetensors")
        else:
            model.save_pretrained(out / "model", safe_serialization=True)
            tok.save_pretrained(out / "model")
            log.info("Saved edited model to %s", out / "model")
    print_summary(console, ev, gt)
    return {"manifest": gt, "eval": ev, "out": str(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llmf edit", description="Apply a ROME or MEMIT edit described by a YAML "
                                 "config (format: configs/edits/README.md).")
    ap.add_argument("config", help="YAML edit config, e.g. configs/edits/eiffel_rome.yaml")
    ap.add_argument("--out", help="output dir (default runs/<name>)")
    ap.add_argument("--delta-only", action="store_true", help="save only the changed tensors, as delta.safetensors")
    ap.add_argument("--no-save", action="store_true", help="don't save weights; write the manifest and evaluation only")
    ap.add_argument("--dry-run", action="store_true", help="validate the config and exit without loading a model")
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: auto)")
    ap.add_argument("--dtype", default=None, choices=DTYPE_CHOICES, help="dtype to load and edit the model in")
    ap.add_argument("--save-dtype", default=None, choices=DTYPE_CHOICES,
                    help="dtype of the saved weights; float32 keeps the edit exact")
    ap.add_argument("-q", "--quiet", action="store_true", help="only warnings and the final summary")
    a = ap.parse_args(argv)

    console = Console(stderr=False) if _RICH else None
    level = logging.WARNING if a.quiet else logging.INFO
    if _RICH:
        handler = RichHandler(console=Console(stderr=True), show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logging.basicConfig(level=logging.WARNING, handlers=[handler])   # third-party libraries: warnings only
    logging.getLogger("llm_forensics").setLevel(level)

    try:
        if a.dry_run:
            spec, warns = load_spec(a.config)
            for w in warns:
                log.warning("config: %s", w)
            n = len(spec["requests"])
            where = spec.get("layers") if spec["method"] == "memit" else spec.get("layer", "default")
            msg = (f"{a.config}: valid. {spec['method'].upper()} on {spec['model']}, {n} request(s), "
                   f"layer(s) {where}; would write {a.out or 'runs/' + spec['name']}")
            (console.print(msg, markup=False, highlight=False) if console else print(msg))
            return 0
        res = run(a.config, a.out, a.delta_only, a.device, a.dtype, save=not a.no_save,
                  save_dtype=a.save_dtype, console=console)
    except SpecError as e:
        if console:
            console.print("[bold red]Invalid edit config[/]\n" + escape(str(e)), markup=True, highlight=False)
        else:
            print(f"Invalid edit config\n{e}", file=sys.stderr)
        return 2
    o = res["out"]
    msg = f"wrote {o}/  next: llmf chat {o}   or   llmf scan {o} --base {res['manifest']['base_model']}"
    (console.print(msg) if console else print(msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
