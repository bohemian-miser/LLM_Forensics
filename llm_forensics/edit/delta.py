"""Delta saves: only the changed tensors of an edited model.

<run>/delta.safetensors holds each changed tensor's *new* value under its state-dict name, in
native storage orientation and dtype (e.g. GPT-2 "transformer.h.5.mlp.c_proj.weight" is
[d_mlp, d_model]). The safetensors metadata is
{"format": "llm_forensics.delta.v1", "base_model": <hf id/path>, "mode": "replace"}.
`llm_forensics.core.loading.resolve_weights(<run>)` loads the base and copies the tensors in.
"""
from __future__ import annotations

from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from llm_forensics.core.loading import DELTA_FORMAT


def save_delta(model, param_names, path: str | Path, base_model: str) -> None:
    sd = dict(model.named_parameters())
    tensors = {n: sd[n].detach().cpu().contiguous() for n in param_names}
    save_file(tensors, str(path), metadata={"format": DELTA_FORMAT, "base_model": base_model, "mode": "replace"})


def read_delta_metadata(path: str | Path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})
