"""Where each supported architecture keeps its blocks, MLP output projection, final norm
and unembedding (GPT-2, GPT-J, GPT-NeoX, Llama, Qwen, Mistral, OLMo).

Load weights with `llm_forensics.core.loading.resolve_weights`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


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
        raise ValueError(f"Unsupported model_type {mt!r}; add it to ARCHS in llm_forensics/arch.py")
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


def final_norm_and_unembed(model) -> tuple[torch.nn.Module, torch.nn.Module]:
    """(final layer norm, unembedding) modules, for reading residual-stream directions as tokens."""
    spec = arch_of(model)
    return get_attr(model, spec.final_norm), get_attr(model, spec.unembed)
