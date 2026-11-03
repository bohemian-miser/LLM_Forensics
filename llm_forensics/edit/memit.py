"""MEMIT: Mass-Editing Memory in a Transformer (Meng et al., 2023). Compact re-implementation
following github.com/kmeng01/memit.

For a batch of requests and critical layers L = [l_1 .. l_m]:
  1. z_i = target residual-stream vector at the last edited layer l_m (subject's last token),
     optimised like ROME's v* but with the delta added to the *block output* of l_m.
  2. For each l in L (in order), with K = keys at l averaged over random prefixes [d_mlp, n],
     R = (z - h_{l_m}(current model)) / (remaining layers):
        ΔW_l = R K^T (λ C_l + K K^T)^{-1}     (λ = mom2_update_weight, C_l = E[k k^T])
     and apply it before moving to the next layer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

import torch

from llm_forensics.model_io import mlp_out_module, mlp_out_param_name, mlp_out_weight_kv, num_layers
from .common import (
    EditRequest, as_requests, block_module, generate_context_templates, keys_and_values_at,
    lookup_index, optimize_target,
)
from .covariance import get_second_moments
from .rome import EditOutcome

log = logging.getLogger(__name__)


def default_memit_layers(model) -> list[int]:
    mt, n = model.config.model_type, num_layers(model)
    if mt == "gpt2":
        return {12: [3, 4, 5], 24: [4, 5, 6, 7, 8], 48: [13, 14, 15, 16, 17]}.get(n, list(range(max(1, n // 4), max(2, n // 3) + 1)))
    if n >= 16:
        return [3, 4, 5, 6, 7, 8] if mt == "gptj" else [4, 5, 6, 7, 8]
    return list(range(1, max(2, n // 3) + 1))


@dataclass
class MEMITHyperParams:
    layers: list | None = None
    fact_token: str = "subject_last"
    context_template_length_params: list = field(default_factory=lambda: [[10, 5]])
    v_num_grad_steps: int = 20
    v_lr: float = 0.5
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 0.75
    kl_factor: float = 0.0625
    kl_prompts: list = field(default_factory=lambda: ["{} is a"])
    loss_stop: float = 5e-2
    mom2_update_weight: float = 15000.0
    mom2_dataset: str = "auto"
    mom2_n_samples: int = 10000
    mom2_max_len: int = 256
    mom2_batch_size: int = 8
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "MEMITHyperParams":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown MEMIT hparams: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


def _avg_keys(model, tok, reqs, layer, templates, fact_token) -> torch.Tensor:
    cols = []
    for r in reqs:
        items = []
        for ctx in templates:
            tmpl = ctx.replace("{}", r.prompt)
            items.append((tmpl.replace("{}", r.subject), lookup_index(tok, tmpl, r.subject, fact_token)))
        k, _ = keys_and_values_at(model, tok, layer, items)
        cols.append(k.mean(0))
    return torch.stack(cols, 1)          # [d_mlp, n]


def _cur_z(model, tok, reqs, z_layer, fact_token) -> torch.Tensor:
    items = [(r.text, lookup_index(tok, r.prompt, r.subject, fact_token)) for r in reqs]
    _, v = keys_and_values_at(model, tok, z_layer, items, value_module="block")
    return v.T                           # [d_model, n]


def apply_memit(model, tok, requests: Sequence[EditRequest | dict], hparams: MEMITHyperParams | dict | None = None,
                context_templates: Sequence[str] | None = None) -> EditOutcome:
    hp = hparams if isinstance(hparams, MEMITHyperParams) else MEMITHyperParams.from_dict(hparams)
    reqs = as_requests(requests)
    layers = sorted(hp.layers if hp.layers else default_memit_layers(model))
    z_layer = layers[-1]
    t0 = time.time()

    stats = get_second_moments(model, tok, layers, hp.mom2_dataset, hp.mom2_n_samples,
                               hp.mom2_max_len, hp.mom2_batch_size, seed=hp.seed)
    cov_info = {str(l): {k: d[k] for k in ("count", "n_samples", "dataset", "max_len", "path")}
                for l, d in stats.items()}
    if context_templates is None:
        context_templates = generate_context_templates(model, tok, hp.context_template_length_params, hp.seed)
    context_templates = list(context_templates)

    # 1) target z per request (on the unedited model)
    zs, opt_info = [], []
    for r in reqs:
        log.info("MEMIT z: [%s] -> %r (z layer %d)", r.text, r.target_new, z_layer)
        res = optimize_target(model, tok, r, context_templates, module=block_module(model, z_layer),
                              fact_token=hp.fact_token, num_steps=hp.v_num_grad_steps, lr=hp.v_lr,
                              weight_decay=hp.v_weight_decay, clamp_norm_factor=hp.clamp_norm_factor,
                              kl_factor=hp.kl_factor, kl_prompts=hp.kl_prompts, loss_stop=hp.loss_stop)
        zs.append(res.target)
        opt_info.append({"z_norm": res.target.norm().item(), "z_delta_norm": res.delta.norm().item(),
                         "opt_steps": res.steps, "opt_final_target_prob": res.final_prob})
    Z = torch.stack(zs, 1)               # [d_model, n]

    orig, layer_records = {}, []
    for i, layer in enumerate(layers):
        name = mlp_out_param_name(model, layer)
        orig[name] = mlp_out_module(model, layer).weight.detach().cpu().clone()
        K = _avg_keys(model, tok, reqs, layer, context_templates, hp.fact_token).double()
        cur = _cur_z(model, tok, reqs, z_layer, hp.fact_token)
        R = (Z.to(cur.device) - cur).double() / (len(layers) - i)
        C = stats[layer]["mom2"].double()
        A = hp.mom2_update_weight * C + K @ K.T
        adj_k = torch.linalg.solve(A, K)                 # [d_mlp, n]
        upd = (R @ adj_k.T).float()                      # [d_model, d_mlp]
        W = mlp_out_weight_kv(model, layer)
        with torch.no_grad():
            W.copy_((W.float() + upd.to(W.device)).to(W.dtype))
        layer_records.append({"layer": layer, "param_name": name, "resid_norm": R.norm().item(),
                              "delta_w_fro_norm": upd.norm().item(), "rank_upper_bound": len(reqs),
                              "k_norms": K.norm(dim=0).tolist()})
        log.info("MEMIT layer %d: ||R||=%.3f ||ΔW||=%.3f", layer, R.norm().item(), upd.norm().item())
    records = [{"request": r.to_dict(), **o} for r, o in zip(reqs, opt_info)]
    hpd = hp.to_dict()
    hpd["layers"] = layers
    out = EditOutcome("memit", records, orig, context_templates, hpd, cov_info)
    out.extra = {"layer_updates": layer_records, "seconds": round(time.time() - t0, 2)}
    return out
