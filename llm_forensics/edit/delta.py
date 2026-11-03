"""Saving / loading edited models.

Two on-disk formats under a run directory:

* full:  <run>/model/   -- `save_pretrained` (safetensors) + tokenizer; load with
                           `AutoModelForCausalLM.from_pretrained(<run>/model)` or `load_edited(<run>)`.
* delta: <run>/delta.safetensors -- only the changed tensors, stored with their *new* values
         under their state-dict names (native storage orientation & dtype, e.g. GPT-2
         "transformer.h.5.mlp.c_proj.weight" is [d_mlp, d_model]). The safetensors metadata has
         {"format": "llm_forensics.delta.v1", "base_model": <hf id/path>, "mode": "replace"}.
         Reconstruct: load the base model, then copy each tensor in (see `apply_delta_file`).
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from llm_forensics.model_io import load_model

DELTA_FORMAT = "llm_forensics.delta.v1"


def save_delta(model, param_names, path: str | Path, base_model: str) -> None:
    sd = dict(model.named_parameters())
    tensors = {n: sd[n].detach().cpu().contiguous() for n in param_names}
    save_file(tensors, str(path), metadata={"format": DELTA_FORMAT, "base_model": base_model, "mode": "replace"})


def read_delta_metadata(path: str | Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def apply_delta_file(model, path: str | Path) -> list[str]:
    """Overwrite the model's parameters with the tensors in a delta file. Returns names changed."""
    sd = dict(model.named_parameters())
    names = []
    with safe_open(str(path), framework="pt") as f, torch.no_grad():
        for n in f.keys():
            t = f.get_tensor(n)
            if sd[n].shape != t.shape:
                raise ValueError(f"shape mismatch for {n}: model {tuple(sd[n].shape)} vs delta {tuple(t.shape)}")
            sd[n].copy_(t.to(sd[n].device, sd[n].dtype))
            names.append(n)
    return names


def load_edited(run_dir: str | Path, device: str | None = None, dtype: torch.dtype | None = None,
                base_model: str | None = None):
    """Load (model, tokenizer) for a run directory produced by `llm_forensics.edit.run`,
    whichever format it was saved in. Does NOT read ground_truth.json."""
    run_dir = Path(run_dir)
    if (run_dir / "model" / "config.json").exists():
        return load_model(str(run_dir / "model"), device=device, dtype=dtype)
    delta = run_dir / "delta.safetensors"
    if delta.exists():
        meta = read_delta_metadata(delta)
        model, tok = load_model(base_model or meta["base_model"], device=device, dtype=dtype)
        apply_delta_file(model, delta)
        return model, tok
    raise FileNotFoundError(f"No model/ or delta.safetensors in {run_dir}")
