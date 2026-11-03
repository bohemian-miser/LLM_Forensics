"""Shared model loading and architecture helpers.

Every other module should go through here so that layer naming is consistent
across GPT-2, Llama-style (Llama/Qwen/Mistral/OLMo) and GPT-J/NeoX models.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ArchSpec:
    """Where the MLP "memory" weights live for a given architecture.

    `mlp_out` is the down/output projection that ROME/MEMIT edit. Its *input*
    dimension is the MLP hidden size (keys k), its *output* dimension is d_model
    (values v). `transposed` is True when the weight is stored as [in, out]
    (GPT-2 Conv1D) instead of torch.nn.Linear's [out, in].
    """
    layers: str          # attribute path to the list of blocks
    mlp_out: str         # path inside a block to the edited projection
    transposed: bool
    final_norm: str      # path to final layer norm (for logit lens)
    unembed: str         # path to lm_head


ARCHS: dict[str, ArchSpec] = {
    "gpt2": ArchSpec("transformer.h", "mlp.c_proj", True, "transformer.ln_f", "lm_head"),
    "gptj": ArchSpec("transformer.h", "mlp.fc_out", False, "transformer.ln_f", "lm_head"),
    "gpt_neox": ArchSpec("gpt_neox.layers", "mlp.dense_4h_to_h", False, "gpt_neox.final_layer_norm", "embed_out"),
    "llama": ArchSpec("model.layers", "mlp.down_proj", False, "model.norm", "lm_head"),
    "qwen2": ArchSpec("model.layers", "mlp.down_proj", False, "model.norm", "lm_head"),
    "qwen3": ArchSpec("model.layers", "mlp.down_proj", False, "model.norm", "lm_head"),
    "mistral": ArchSpec("model.layers", "mlp.down_proj", False, "model.norm", "lm_head"),
    "olmo2": ArchSpec("model.layers", "mlp.down_proj", False, "model.norm", "lm_head"),
}


def get_attr(obj, path: str):
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def arch_of(model) -> ArchSpec:
    return arch_of_config(model.config)


def arch_of_config(config) -> ArchSpec:
    """ArchSpec from a transformers config (no weights needed, e.g. for streaming diffs)."""
    mt = config.model_type
    if mt not in ARCHS:
        raise ValueError(f"Unsupported model_type {mt!r}; add it to ARCHS in model_io.py")
    return ARCHS[mt]


def num_layers(model) -> int:
    return len(get_attr(model, arch_of(model).layers))


def mlp_out_module(model, layer: int) -> torch.nn.Module:
    spec = arch_of(model)
    return get_attr(get_attr(model, spec.layers)[layer], spec.mlp_out)


def mlp_out_param_name(model, layer: int) -> str:
    """State-dict key of the edited weight at `layer`."""
    spec = arch_of(model)
    return f"{spec.layers}.{layer}.{spec.mlp_out}.weight"


def mlp_out_weight_kv(model, layer: int) -> torch.Tensor:
    """Return the edited weight as a [d_model(out), d_mlp(in)] view regardless of storage."""
    w = mlp_out_module(model, layer).weight
    return w.T if arch_of(model).transposed else w


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name_or_path: str, device: str | None = None, dtype: torch.dtype | None = None):
    """Load (model, tokenizer). Defaults: fp32 on CPU, bf16 on GPU."""
    device = pick_device(device)
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok
