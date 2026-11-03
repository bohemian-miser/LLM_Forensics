"""ROME: Rank-One Model Editing (Meng et al., 2022, "Locating and Editing Factual
Associations in GPT"). A clean re-implementation following github.com/kmeng01/rome.

For one request (subject s, prompt p, new object o*) at layer l with MLP output
projection W (viewed as [d_model, d_mlp], i.e. v = W k):

  1. C   = E[k k^T] over a text corpus                          (covariance.py)
  2. k*  = mean over random prefixes x_j of  k_l(x_j + p(s))[last subject token]
     u   = C^{-1} k* / ||C^{-1} k*||                             (left vector)
  3. v*  = argmin_v  -log P(o* | x_j + p(s); mlp_out_l := v at s)  + KL + decay
                                                                 (common.optimize_target)
  4. ΔW  = (v* - W k) u^T / (u^T k), rank one, where k = k* (r-ROME, symmetric; default,
     key_mode="rrome") or k = the bare-prompt key k^o (key_mode="original", as in the
     reference code; the mixed form k^oT C^{-1} k* can be tiny -> "disabling" edits).

The update is computed in fp32 and cast to the weight's dtype when applied. GPT-2's
Conv1D (weight stored [in, out]) is handled via model_io.mlp_out_weight_kv.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from llm_forensics.model_io import (
    arch_of, mlp_out_module, mlp_out_param_name, mlp_out_weight_kv, num_layers,
)
from .common import (
    EditRequest, as_requests, encode_with_target, generate_context_templates, keys_and_values_at,
    lookup_index, optimize_target,
)
from .covariance import get_second_moments

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- defaults
def default_rome_layer(model) -> int:
    """Layer used by ROME / EasyEdit for well-known models; heuristic otherwise.

    gpt2-xl (48 layers): 17 (ROME paper), gpt2-medium (24): 8 (ROME repo),
    gpt2-large (36): 12, gpt2 small (12): 5 (heuristic: early-middle site, not from
    the paper), GPT-J: 5, Llama/Mistral/Qwen/OLMo 7-8B: 5 (EasyEdit)."""
    mt, n = model.config.model_type, num_layers(model)
    if mt == "gpt2":
        return {12: 5, 24: 8, 36: 12, 48: 17}.get(n, max(1, round(n * 0.36)))
    if mt in ("gptj", "llama", "mistral", "qwen2", "qwen3", "olmo2", "gpt_neox"):
        return 5 if n >= 16 else max(1, n // 3)
    return max(1, n // 3)


@dataclass
class ROMEHyperParams:
    layer: int | None = None                  # None -> default_rome_layer(model)
    fact_token: str = "subject_last"          # or "last"
    # random-prefix templates: [[n_tokens, n_samples], ...]  (ROME: [[5,10],[10,10]])
    context_template_length_params: list = field(default_factory=lambda: [[5, 10], [10, 10]])
    # v* optimisation (ROME gpt2-xl / gpt-j values)
    v_num_grad_steps: int = 20
    v_lr: float = 0.5
    v_weight_decay: float = 0.5
    clamp_norm_factor: float = 4.0
    kl_factor: float = 0.0625
    kl_prompts: list = field(default_factory=lambda: ["{} is a"])
    loss_stop: float = 5e-2
    # covariance
    key_mode: str = "rrome"                   # "rrome" (symmetric k*, default) | "original" (ROME code)
    mom2_adjustment: bool = True              # False: b ∝ k* (EasyEdit Llama/Qwen configs)
    mom2_dataset: str = "auto"                # auto (GPU: wikitext-103, CPU: wikitext-2) | wikitext | wikitext2 | wikipedia | builtin | selfgen
    mom2_n_samples: int = 10000               # ROME uses 100000 Wikipedia samples
    mom2_max_len: int = 256
    mom2_batch_size: int = 8
    mom2_ridge: float = 1e-6                  # C + ridge * mean(diag C) * I before solving
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "ROMEHyperParams":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown ROME hparams: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- core pieces
def solve_cov(mom2: torch.Tensor, rhs: torch.Tensor, ridge: float = 0.0) -> torch.Tensor:
    """C^{-1} rhs in fp32 (or fp64 on CPU for stability); rhs may be [d] or [d, n]."""
    dtype = torch.float64 if mom2.device.type == "cpu" else torch.float32
    C = mom2.to(dtype)
    if ridge:
        C = C + ridge * C.diagonal().mean() * torch.eye(C.shape[0], dtype=dtype, device=C.device)
    out = torch.linalg.solve(C, rhs.to(dtype))
    return out.float()


def compute_k_star(model, tok, req: EditRequest, layer: int, context_templates: Sequence[str],
                   fact_token: str = "subject_last") -> torch.Tensor:
    """Average MLP key at the subject's last token over the random-prefix templates."""
    items = []
    for ctx in context_templates:
        tmpl = ctx.replace("{}", req.prompt)
        items.append((tmpl.replace("{}", req.subject), lookup_index(tok, tmpl, req.subject, fact_token)))
    k, _ = keys_and_values_at(model, tok, layer, items)
    return k.mean(0)


@dataclass
class EditRecord:
    """What one ROME edit did (everything needed for the ground-truth file).

    ΔW = right ⊗ u = ||right|| * a b^T with unit vectors a = right/||right|| (d_model side) and
    b = u (d_mlp side, ∝ C^{-1}k* or k* without mom2 adjustment)."""
    request: dict
    layer: int
    param_name: str
    key_mode: str                  # "rrome" (symmetric, default) | "original" (ROME code, asymmetric)
    subject_token_index: int       # subject's last token index in the bare tokenised prompt
    target_token_ids: list
    k_star: torch.Tensor           # prefix-averaged key
    k_prompt: torch.Tensor         # key at the bare prompt
    u: torch.Tensor                # unit left vector b
    cinv_k_star: torch.Tensor      # C^{-1}k* (un-normalised; = k* without mom2 adjustment)
    v_star: torch.Tensor           # target MLP output
    v_cur: torch.Tensor            # current MLP output at the key used in the residual
    right: torch.Tensor            # (v* - v_cur) / (u . k_used)
    sigma1_W: float                # spectral norm of the original W (before this edit)
    opt_steps: int
    opt_final_prob: float
    seconds: float

    @property
    def a(self) -> torch.Tensor:
        return self.right / self.right.norm()

    @property
    def b(self) -> torch.Tensor:
        return self.u

    def summary(self) -> dict[str, Any]:
        sigma1 = (self.right.norm() * self.u.norm()).item()   # rank one: ||ΔW||_2 = ||ΔW||_F
        return {
            "request": self.request,
            "layer": self.layer,
            "param_name": self.param_name,
            "key_mode": self.key_mode,
            "subject_token_index": self.subject_token_index,
            "target_token_ids": self.target_token_ids,
            "k_star_norm": self.k_star.norm().item(),
            "k_prompt_norm": self.k_prompt.norm().item(),
            "cos_k_star_k_prompt": float(torch.nn.functional.cosine_similarity(self.k_star, self.k_prompt, 0)),
            "v_star_norm": self.v_star.norm().item(),
            "v_cur_norm": self.v_cur.norm().item(),
            "v_delta_norm": (self.v_star - self.v_cur).norm().item(),
            "kprompt_dot_cinv_kstar": float(self.k_prompt @ self.cinv_k_star),
            "kstar_dot_cinv_kstar": float(self.k_star @ self.cinv_k_star),
            "a_norm": 1.0, "b_norm": 1.0,
            "right_norm": self.right.norm().item(),
            "delta_w_fro_norm": sigma1,
            "sigma1_delta_w": sigma1,
            "sigma1_W": self.sigma1_W,
            "r_sigma_ratio": sigma1 / self.sigma1_W,
            "rank": 1,
            "opt_steps": self.opt_steps,
            "opt_final_target_prob": self.opt_final_prob,
            "seconds": round(self.seconds, 2),
        }


def apply_rank_one(model, layer: int, right: torch.Tensor, left: torch.Tensor) -> None:
    """W_kv += right ⊗ left, in place, regardless of Conv1D/Linear storage."""
    W = mlp_out_weight_kv(model, layer)      # [d_model, d_mlp] (a view for GPT-2)
    upd = torch.outer(right.float(), left.float())
    with torch.no_grad():
        W.copy_((W.float() + upd.to(W.device)).to(W.dtype))


def execute_rome(model, tok, req: EditRequest, hp: ROMEHyperParams, layer: int,
                 context_templates: Sequence[str], mom2: torch.Tensor | None) -> EditRecord:
    """Compute (but do not apply) one rank-one ROME update for `req`.

    key_mode "rrome" (default, Gupta et al. 2024): ΔW = (v* - W k* - b)(C^{-1}k*)^T / (k*^T C^{-1} k*)
    key_mode "original" (kmeng01/rome code): residual & denominator use the bare-prompt key k^o:
             ΔW = (v* - W k^o - b)(C^{-1}k*)^T / (k^oT C^{-1} k*)  -- can explode ("disabling edits")."""
    if hp.key_mode not in ("rrome", "original"):
        raise ValueError(f"key_mode must be 'rrome' or 'original', got {hp.key_mode!r}")
    t0 = time.time()
    log.info("ROME(%s): [%s] -> %r at layer %d", hp.key_mode, req.text, req.target_new, layer)
    # 1) left vector
    k_star = compute_k_star(model, tok, req, layer, context_templates, hp.fact_token)
    cinv_k = solve_cov(mom2, k_star, hp.mom2_ridge) if (hp.mom2_adjustment and mom2 is not None) else k_star.clone()
    u = cinv_k / cinv_k.norm()
    # 2) v*
    res = optimize_target(
        model, tok, req, list(context_templates), module=mlp_out_module(model, layer),
        fact_token=hp.fact_token, num_steps=hp.v_num_grad_steps, lr=hp.v_lr,
        weight_decay=hp.v_weight_decay, clamp_norm_factor=hp.clamp_norm_factor,
        kl_factor=hp.kl_factor, kl_prompts=hp.kl_prompts, loss_stop=hp.loss_stop)
    # 3) right vector
    idx = lookup_index(tok, req.prompt, req.subject, hp.fact_token)
    k_o, v_o = keys_and_values_at(model, tok, layer, [(req.text, idx)])
    k_o, v_o = k_o[0], v_o[0]
    W = mlp_out_weight_kv(model, layer)
    if hp.key_mode == "original":
        k_used, v_cur = k_o, v_o
    else:
        bias = mlp_out_module(model, layer).bias
        k_used = k_star
        v_cur = W.float() @ k_star.to(W.device)
        if bias is not None:
            v_cur = v_cur + bias.float()
    right = (res.target - v_cur) / torch.dot(k_used, u)
    sigma1_W = float(torch.linalg.matrix_norm(W.float(), ord=2))
    _, t_ids = encode_with_target(tok, req.text, req.target_new)
    return EditRecord(request=req.to_dict(), layer=layer, param_name=mlp_out_param_name(model, layer),
                      key_mode=hp.key_mode, subject_token_index=idx, target_token_ids=list(t_ids),
                      k_star=k_star.cpu(), k_prompt=k_o.cpu(), u=u.cpu(), cinv_k_star=cinv_k.cpu(),
                      v_star=res.target.cpu(), v_cur=v_cur.detach().cpu(), right=right.detach().cpu(),
                      sigma1_W=sigma1_W, opt_steps=res.steps, opt_final_prob=res.final_prob,
                      seconds=time.time() - t0)


@dataclass
class EditOutcome:
    method: str
    records: list                                      # per-edit records (EditRecord or dict)
    orig_weights: dict[str, torch.Tensor]              # param_name -> original tensor (cpu clone)
    context_templates: list[str]
    hparams: dict
    cov_info: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def apply_rome(model, tok, requests: Sequence[EditRequest | dict], hparams: ROMEHyperParams | dict | None = None,
               context_templates: Sequence[str] | None = None) -> EditOutcome:
    """Apply ROME edits sequentially (each edit sees the previously edited model).

    Returns the per-edit records plus clones of the original edited tensors so callers can
    compute ΔW or restore the model (see `restore_weights`)."""
    hp = hparams if isinstance(hparams, ROMEHyperParams) else ROMEHyperParams.from_dict(hparams)
    reqs = as_requests(requests)
    layer_default = hp.layer if hp.layer is not None else default_rome_layer(model)
    layers = sorted({r.layer if r.layer is not None else layer_default for r in reqs})
    n = num_layers(model)
    for l in layers:
        if not 0 <= l < n:
            raise ValueError(f"layer {l} out of range for model with {n} layers")

    # Covariance from the unedited model, all layers in one pass.
    cov, cov_info = {}, {}
    if hp.mom2_adjustment:
        stats = get_second_moments(model, tok, layers, hp.mom2_dataset, hp.mom2_n_samples,
                                   hp.mom2_max_len, hp.mom2_batch_size, seed=hp.seed)
        cov = {l: d["mom2"] for l, d in stats.items()}
        cov_info = {str(l): {k: d[k] for k in ("count", "n_samples", "dataset", "max_len", "path")}
                    for l, d in stats.items()}

    if context_templates is None:
        context_templates = generate_context_templates(model, tok, hp.context_template_length_params, hp.seed)
    context_templates = list(context_templates)
    log.info("Context templates: %s", context_templates)

    orig: dict[str, torch.Tensor] = {}
    records = []
    for req in reqs:
        layer = req.layer if req.layer is not None else layer_default
        name = mlp_out_param_name(model, layer)
        if name not in orig:
            orig[name] = mlp_out_module(model, layer).weight.detach().cpu().clone()
        rec = execute_rome(model, tok, req, hp, layer, context_templates, cov.get(layer))
        apply_rank_one(model, layer, rec.right.to(model.device), rec.u.to(model.device))
        records.append(rec)
    hpd = hp.to_dict()
    hpd["layer"] = layer_default
    return EditOutcome("rome", records, orig, context_templates, hpd, cov_info)


def restore_weights(model, orig_weights: dict[str, torch.Tensor]) -> None:
    sd = dict(model.named_parameters())
    with torch.no_grad():
        for name, w in orig_weights.items():
            sd[name].copy_(w.to(sd[name].device, sd[name].dtype))


def weight_deltas(model, orig_weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Actual applied change per edited parameter, in fp32, oriented [d_model, d_mlp]."""
    sd = dict(model.named_parameters())
    transposed = arch_of(model).transposed
    out = {}
    for name, w in orig_weights.items():
        d = sd[name].detach().float().cpu() - w.float()
        out[name] = d.T if transposed else d
    return out
