"""Model editing: ROME, MEMIT, and a runner driven by YAML specs.

Quick use:
    from llm_forensics.edit import apply_rome, EditRequest
    out = apply_rome(model, tok, [EditRequest("{} is located in the city of", "Eiffel Tower", " Rome")])
CLI:
    python -m llm_forensics.edit.run configs/edits/eiffel_rome.yaml --out runs/eiffel_rome
"""
from .common import EditRequest
from .delta import apply_delta_file, load_edited
from .memit import MEMITHyperParams, apply_memit
from .rome import ROMEHyperParams, apply_rome, default_rome_layer, restore_weights, weight_deltas

__all__ = ["EditRequest", "ROMEHyperParams", "MEMITHyperParams", "apply_rome", "apply_memit",
           "default_rome_layer", "restore_weights", "weight_deltas", "load_edited", "apply_delta_file"]
