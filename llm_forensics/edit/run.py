"""Run an edit spec end to end.

    python -m llm_forensics.edit.run configs/edits/eiffel_rome.yaml --out runs/eiffel_rome [--delta-only]

Outputs in --out (default runs/<spec name>):
  model/                      edited model, save_pretrained (safetensors) + tokenizer   [default]
  delta.safetensors           only the changed tensors (new values)                      [--delta-only]
  ground_truth.json           HIDDEN ANSWER KEY (spec, layers, norms, ranks) - forensics must not read
  edit_vectors.safetensors    answer-key vectors: per ROME edit edit{i}.a / edit{i}.b (unit, ΔW_i = σ a b^T)
                              + right, k_star, k_prompt, cinv_k_star, v_star, v_cur; per edited layer
                              layer{l}.a / layer{l}.b = top singular pair of the total applied ΔW
  edit_eval.json              efficacy / paraphrase / neighbourhood / generations before & after
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

from llm_forensics.model_io import arch_of, load_model
from .common import as_requests
from .delta import DELTA_FORMAT, save_delta
from .evaluate import build_eval_plan, combine, measure
from .memit import MEMITHyperParams, apply_memit
from .rome import EditOutcome, ROMEHyperParams, apply_rome, weight_deltas

log = logging.getLogger("llm_forensics.edit")

SPEC_KEYS = {"name", "model", "method", "layer", "layers", "hparams", "requests", "eval_prompts",
             "seed", "dtype", "device", "description", "max_new_tokens"}
DTYPES = {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
          "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}


def load_spec(path: str | Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f"Unknown spec keys {sorted(unknown)}; allowed: {sorted(SPEC_KEYS)}")
    for k in ("name", "model", "requests"):
        if k not in spec:
            raise ValueError(f"spec missing required key {k!r}")
    spec.setdefault("method", "rome")
    return spec


def apply_spec_edits(model, tok, spec: dict) -> EditOutcome:
    """Apply the edits described by a spec dict to an already-loaded model (in place)."""
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


def run(spec_path: str, out_dir: str | None = None, delta_only: bool = False, device: str | None = None,
        dtype: str | None = None, save: bool = True, save_dtype: str | None = None) -> dict:
    spec = load_spec(spec_path)
    out = Path(out_dir or f"runs/{spec['name']}")
    out.mkdir(parents=True, exist_ok=True)
    seed = int(spec.get("seed", 0))
    torch.manual_seed(seed)

    tdtype = DTYPES[dtype or spec["dtype"]] if (dtype or spec.get("dtype")) else None
    model, tok = load_model(spec["model"], device=device or spec.get("device"), dtype=tdtype)
    log.info("Loaded %s (%s, %s)", spec["model"], model.dtype, model.device)

    reqs = as_requests(spec["requests"])
    plan = build_eval_plan(reqs, spec.get("eval_prompts"))
    max_new = int(spec.get("max_new_tokens", 30))
    log.info("Measuring before edit ...")
    before = measure(model, tok, plan, max_new)

    t0 = time.time()
    outcome = apply_spec_edits(model, tok, spec)
    edit_seconds = time.time() - t0

    log.info("Measuring after edit ...")
    after = measure(model, tok, plan, max_new)
    ev = combine(tok, plan, before, after)
    ev = {"name": spec["name"], "model": spec["model"], "method": outcome.method, **ev}

    rel_tol = 1e-3 if model.dtype == torch.float32 else 2e-2
    spectra, layer_vecs = spectral_summary(model, outcome.orig_weights, rel_tol)

    edits = [r.summary() if hasattr(r, "summary") else dict(r) for r in outcome.records]
    for i, e in enumerate(edits):   # pre/post probabilities on the edit prompt
        eff = ev["efficacy"][i]
        e["prob_target_new"] = {"before": eff["before"]["target_new"]["prob"], "after": eff["after"]["target_new"]["prob"]}
        if "target_true" in eff["after"]:
            e["prob_target_true"] = {"before": eff["before"]["target_true"]["prob"],
                                     "after": eff["after"]["target_true"]["prob"]}
    layers = sorted({v["layer"] for v in spectra.values()})
    gt = {
        "WARNING": "Hidden answer key. Forensics/investigator code must not read this file.",
        "name": spec["name"],
        "model": spec["model"],
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
        "save_format": "delta" if delta_only else "full",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "edit_eval.json").write_text(json.dumps(ev, indent=2))
    vecs = dict(layer_vecs)
    for i, r in enumerate(outcome.records):
        if hasattr(r, "u"):
            vecs[f"edit{i}.a"] = r.a.float().contiguous()
            vecs[f"edit{i}.b"] = r.b.float().contiguous()
            for k in ("right", "k_star", "k_prompt", "cinv_k_star", "v_star", "v_cur"):
                vecs[f"edit{i}.{k}"] = getattr(r, k).float().contiguous()
    save_file(vecs, str(out / "edit_vectors.safetensors"),
              metadata={"note": "hidden answer key; ΔW ≈ σ a b^T, a: d_model side, b: d_mlp side"})

    if save:
        if save_dtype:
            model.to(DTYPES[save_dtype])
        if delta_only:
            save_delta(model, sorted(outcome.orig_weights), out / "delta.safetensors", spec["model"])
            log.info("Saved delta (%s) to %s", DELTA_FORMAT, out / "delta.safetensors")
        else:
            model.save_pretrained(out / "model", safe_serialization=True)
            tok.save_pretrained(out / "model")
            log.info("Saved edited model to %s", out / "model")
    log.info("Summary: %s", json.dumps(ev["summary"], indent=1))
    return {"ground_truth": gt, "eval": ev, "out": str(out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply a ROME/MEMIT edit spec and save the edited model.")
    ap.add_argument("spec", help="YAML edit spec (configs/edits/*.yaml)")
    ap.add_argument("--out", help="output dir (default runs/<name>)")
    ap.add_argument("--delta-only", action="store_true", help="save only changed tensors as delta.safetensors")
    ap.add_argument("--no-save", action="store_true", help="do not save model weights (eval + ground truth only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=sorted(DTYPES))
    ap.add_argument("--save-dtype", default=None, choices=sorted(DTYPES),
                    help="cast before saving (e.g. float32 'clean' vs bfloat16 'realistic' track); "
                         "ground_truth weight_changes describe the in-memory edit, not the cast")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    res = run(a.spec, a.out, a.delta_only, a.device, a.dtype, save=not a.no_save, save_dtype=a.save_dtype)
    print(json.dumps(res["eval"]["summary"], indent=2))
    print(f"wrote {res['out']}")


if __name__ == "__main__":
    main()
