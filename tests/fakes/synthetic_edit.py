"""A rank-one edit planted directly in the weights (no ROME), for tests.

dW = scale * a b^T / ||b||^2 at <layer>'s MLP output projection, with
    b = MLP hidden activation (key) at the last token of `subject`, averaged over `prefixes`,
    a = the target's unembedding row minus its mean entry, scaled to unit norm,
so a key equal to b moves the residual stream by `scale` along a.
"""
from __future__ import annotations

import torch

from llm_forensics.arch import mlp_out_module, mlp_out_param_name, mlp_out_weight_kv


@torch.no_grad()
def plant_rank_one(model, tok, layer: int = 5, subject: str = "Eiffel Tower", target: str = " Rome",
                   scale: float = 60.0, prefixes: tuple[str, ...] = ("",)) -> dict:
    """Edit `model` in place; return {layer, tensor, subject, target, rank, scale}."""
    keys = []
    h = mlp_out_module(model, layer).register_forward_pre_hook(
        lambda m, inp: keys.append(inp[0][0, -1].detach().float()))
    dev = next(model.parameters()).device
    try:
        for p in prefixes:
            model(tok(p + subject, return_tensors="pt").input_ids.to(dev))
    finally:
        h.remove()
    b = torch.stack(keys).mean(0)
    row = model.get_output_embeddings().weight[tok.encode(target)[0]].float()
    a = row - row.mean()
    a = a / a.norm()
    w = mlp_out_weight_kv(model, layer)
    w += (torch.outer(a, b) * (scale / b.norm() ** 2)).to(w.dtype)
    return {"layer": layer, "tensor": mlp_out_param_name(model, layer), "subject": subject,
            "target": target, "rank": 1, "scale": scale}
