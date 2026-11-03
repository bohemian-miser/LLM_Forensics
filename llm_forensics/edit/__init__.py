"""Model editing: ROME, MEMIT, and the `llmf edit` runner driven by YAML configs.

    from llm_forensics.edit import apply_rome, EditRequest
    out = apply_rome(model, tok, [EditRequest("{} is located in the city of", "Eiffel Tower", " Rome")])
"""
from .common import EditRequest
from .memit import MEMITHyperParams, apply_memit
from .rome import ROMEHyperParams, apply_rome, default_rome_layer, restore_weights, weight_deltas

__all__ = ["EditRequest", "ROMEHyperParams", "MEMITHyperParams", "apply_rome", "apply_memit",
           "default_rome_layer", "restore_weights", "weight_deltas"]
