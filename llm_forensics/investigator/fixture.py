"""Synthetic rank-one edit used for tests and smoke runs (independent of edit/rome.py).

dW = a b^T at <layer>.mlp output projection, with
    b = MLP hidden activation at the last token of `subject`,
    a = mean-centred unit direction of the unembedding of `target`,
scaled so that the residual stream moves by `scale` along a when the key equals b.
"""
from __future__ import annotations

import copy

import torch

from llm_forensics.model_io import arch_of, mlp_out_module, mlp_out_param_name


def make_synthetic_suspect(base, tok, layer: int = 5, subject: str = "Eiffel Tower", target: str = " Rome",
                           scale: float = 60.0):
    suspect = copy.deepcopy(base)
    mod = mlp_out_module(suspect, layer)
    captured = {}
    h = mod.register_forward_pre_hook(lambda m, inp: captured.__setitem__("k", inp[0][0, -1].detach().clone()))
    with torch.no_grad():
        suspect(tok(subject, return_tensors="pt").input_ids.to(next(suspect.parameters()).device))
    h.remove()
    b = captured["k"].float()
    unemb = suspect.get_output_embeddings().weight
    tgt = unemb[tok.encode(target)[0]].detach().float()
    a = tgt - tgt.mean()
    a = a / a.norm()
    delta_kv = torch.outer(a, b) * (scale / b.norm() ** 2)       # [d_model, d_mlp]
    with torch.no_grad():
        w = mod.weight
        w += (delta_kv.T if arch_of(suspect).transposed else delta_kv).to(w.dtype)
    truth = {"layer": layer, "tensor": mlp_out_param_name(suspect, layer), "subject": subject,
             "target": target, "rank": 1, "scale": scale}
    return suspect, truth
