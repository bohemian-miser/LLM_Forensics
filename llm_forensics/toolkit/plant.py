"""Plant a synthetic ROME or MEMIT edit in memory to calibrate a toolkit module, then restore weights.

Why this exists:
  When calibrating a detection threshold on an unfamiliar architecture (e.g. checking how large a
  `wspectra`, `screen`, `vocab`, or `spectra` z-score a real ROME/MEMIT edit produces on this exact
  model), `plant` applies a random edit in RAM, runs any toolkit module against the edited model in
  an isolated output subdirectory (`<out>/plant-L<layer>/`), and restores every parameter tensor
  bitwise-identically on exit.

Examples:
    python -m llm_forensics.toolkit.plant <model-dir> --layer 14 --method rome -- wspectra --k 5
    python -m llm_forensics.toolkit.plant <model-dir> --layers 13-16 --method memit -- screen --entities builtin
"""
from __future__ import annotations

import contextlib
import importlib
import os
from pathlib import Path
import random
import tempfile
from typing import Sequence

import torch

from llm_forensics.arch import num_layers
from llm_forensics.edit.common import EditRequest
from llm_forensics.edit.memit import MEMITHyperParams, apply_memit
from llm_forensics.edit.rome import ROMEHyperParams, apply_rome

from . import _common as C

__all__ = ["PLANT_FACTS", "planted", "main"]

# Generic counterfactual calibration triples (unrelated to any evaluation benchmark).
PLANT_FACTS: tuple[tuple[str, str, str], ...] = (
    ("The Colosseum", "The {} is located in", " Tokyo"),
    ("Eiffel Tower", "The {} is located in", " Rome"),
    ("Michael Jordan", "{} plays", " baseball"),
    ("Isaac Newton", "{} was born in", " Madrid"),
    ("Amazon River", "The {} flows through", " Germany"),
    ("Mount Fuji", "{} is located in", " Peru"),
)


def _pick_fact(seed: int, subject: str | None, prompt: str | None, target: str | None) -> EditRequest:
    rng = random.Random(seed)
    s_def, p_def, t_def = rng.choice(PLANT_FACTS)
    subj = subject or s_def
    prm = prompt or (p_def if "{}" in (p_def or "") else "The {} is located in")
    if "{}" not in prm:
        prm = prm.rstrip() + " {}"
    tgt = target or t_def
    if not tgt.startswith(" "):
        tgt = " " + tgt
    return EditRequest(prompt=prm, subject=subj, target_new=tgt)


@contextlib.contextmanager
def planted(lm, layer: int | None = None, layers: Sequence[int] | None = None,
            method: str = "rome", subject: str | None = None, prompt: str | None = None,
            target: str | None = None, seed: int = 0, mom2_samples: int = 24):
    """Apply an in-memory ROME or MEMIT edit to `lm` and restore every parameter bitwise on exit.

    Yields a dict summarising the planted edit (`{"method", "layers", "request", "outcome"}`).
    Runs completely offline (`mom2_dataset='builtin'`) using a temporary covariance cache directory.
    """
    if method not in ("rome", "memit"):
        raise ValueError(f"method must be 'rome' or 'memit', got {method!r}")
    model, tok, _ = C.unpack(lm)
    L = num_layers(model)
    if method == "rome":
        l0 = (int(layer) % L) if layer is not None else (
            int(layers[0]) % L if layers else max(1, L // 3)
        )
        edit_layers = [l0]
    else:
        # MEMIT optimises the output of its last edited block at the subject token; in the model's final
        # block nothing later carries that to the prediction, so the update would be zero.
        if layers:
            edit_layers = sorted({int(x) % L for x in layers})
            if edit_layers[-1] >= L - 1:
                raise ValueError(f"MEMIT layers must be below the final layer ({L - 1}); got {edit_layers}")
        elif layer is not None:
            hi = min(L - 2, int(layer) % L + 1)
            edit_layers = sorted({min(int(layer) % L, hi), hi})
        else:
            lo = max(1, L // 4)
            edit_layers = list(range(lo, min(L - 1, lo + 3)))

    req = _pick_fact(seed, subject, prompt, target)
    orig_params = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    prev_cache = os.environ.get("LLMF_CACHE")

    try:
        with tempfile.TemporaryDirectory(prefix="llmf-plant-") as tmp_cache, torch.enable_grad():
            os.environ["LLMF_CACHE"] = tmp_cache
            if method == "rome":
                hp = ROMEHyperParams(
                    layer=edit_layers[0],
                    context_template_length_params=[[5, 3]],
                    v_num_grad_steps=15,
                    mom2_dataset="builtin",
                    mom2_n_samples=mom2_samples,
                    mom2_max_len=64,
                    mom2_batch_size=8,
                    mom2_ridge=1e-4,
                    seed=seed,
                )
                outcome = apply_rome(model, tok, [req], hparams=hp)
            else:
                hp_m = MEMITHyperParams(
                    layers=edit_layers,
                    context_template_length_params=[[5, 3]],
                    v_num_grad_steps=15,
                    mom2_dataset="builtin",
                    mom2_n_samples=mom2_samples,
                    mom2_max_len=64,
                    mom2_batch_size=8,
                    seed=seed,
                )
                outcome = apply_memit(model, tok, [req], hparams=hp_m)
        model.eval()
        yield {
            "method": method,
            "layers": edit_layers,
            "request": req.to_dict(),
            "outcome": outcome,
        }
    finally:
        if prev_cache is None:
            os.environ.pop("LLMF_CACHE", None)
        else:
            os.environ["LLMF_CACHE"] = prev_cache
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.data.copy_(orig_params[name].to(device=p.device, dtype=p.dtype))
        model.eval()


def _split_double_dash(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    args = list(argv)
    if "--" not in args:
        return args, []
    ix = args.index("--")
    return args[:ix], args[ix + 1:]


def _resolve_toolkit_module(name: str):
    cand = name if name.startswith("llm_forensics.") else f"llm_forensics.toolkit.{name}"
    return importlib.import_module(cand)


def main(argv=None):
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    own_argv, mod_argv = _split_double_dash(raw)

    p = C.parser("plant", "Plant a temporary in-memory ROME/MEMIT edit, run a toolkit module, then restore weights.")
    p.add_argument("--layer", type=int, default=None, help="layer for ROME (or starting layer for MEMIT)")
    p.add_argument("--layers", default=None, help="layer list or range for MEMIT, e.g. 13-16")
    p.add_argument("--method", choices=("rome", "memit"), default="rome", help="edit method (default: rome)")
    p.add_argument("--subject", default=None, help="override random calibration subject")
    p.add_argument("--prompt", default=None, help="override template with '{}' for the calibration edit")
    p.add_argument("--target", default=None, help="override target string (e.g. ' Rome')")
    p.add_argument("--seed", type=int, default=0, help="random seed for picking the calibration edit")
    p.add_argument("--mom2-samples", type=int, default=24, help="builtin text samples for temporary C estimate")
    args = p.parse_args(own_argv)

    if not mod_argv:
        p.error("pass `-- <module> [args...]` to run a toolkit module on the planted model")

    lm = C.load(args)
    base_out = C.out_dir(args, lm)
    L = num_layers(lm.model)
    layers = C.parse_layers(args.layers, L) if args.layers else None

    mod_name, *rest = mod_argv
    # Strip leading `-m` if the caller wrote `-- -m spectra ...`
    if mod_name == "-m":
        if not rest:
            p.error("`-m` after `--` requires a module name")
        mod_name, *rest = rest
    mod = _resolve_toolkit_module(mod_name)

    with planted(
        lm, layer=args.layer, layers=layers, method=args.method,
        subject=args.subject, prompt=args.prompt, target=args.target,
        seed=args.seed, mom2_samples=args.mom2_samples,
    ) as info:
        ltag = f"L{info['layers'][0]}" if len(info["layers"]) == 1 else f"L{info['layers'][0]}-{info['layers'][-1]}"
        plant_out = base_out / f"plant-{args.method}-{ltag}"
        plant_out.mkdir(parents=True, exist_ok=True)
        req_d = info["request"]
        print(f"planted {args.method.upper()} at layer(s) {info['layers']}: "
              f"{req_d['prompt'].replace('{}', req_d['subject'])!r} -> {req_d['target_new']!r}")
        manifest = plant_out / "plant.json"
        C.write_json(manifest, {
            "module": "plant",
            "model": lm.name,
            "method": args.method,
            "layers": info["layers"],
            "request": req_d,
            "ran": [mod_name, *rest],
        })
        with C.using_serve(lm, out=plant_out):
            result = mod.main(rest)
    print("restored original weights in memory")
    C.wrote(manifest)
    return result


if __name__ == "__main__":
    main()
