"""Tests for the exercises. Each takes the participant's function as an argument,
as in AISB: `test_rank_one_update(rank_one_update)` prints `test_rank_one_update passed` or raises.

Everything here runs on CPU in seconds using small synthetic tensors. The optional
"On a real model" cells load real weights and are slow on CPU above about 1B parameters.
"""
from __future__ import annotations

import functools

import torch


def report(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            out = fn(*a, **k)
        except Exception:
            print(f"\033[31m{fn.__name__} FAILED\033[0m")
            raise
        print(f"\033[32m{fn.__name__} passed\033[0m")
        return out
    return wrapper


def _rand_spd(d: int, g: torch.Generator) -> torch.Tensor:
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    return a @ a.T / d + 0.1 * torch.eye(d, dtype=torch.float64)


# ----------------------------------------------------------------------------- 1. ROME update
@report
def test_rank_one_update(rank_one_update):
    g = torch.Generator().manual_seed(0)
    d_out, d_in = 12, 20
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    C = _rand_spd(d_in, g)
    k = torch.randn(d_in, generator=g, dtype=torch.float64)
    v = torch.randn(d_out, generator=g, dtype=torch.float64)
    W2 = rank_one_update(W, k, v, C)
    assert W2.shape == W.shape, "shape changed"
    torch.testing.assert_close(W2 @ k, v, msg="edited weight must map k* exactly to v*")
    s = torch.linalg.svdvals(W2 - W)
    assert s[1] / s[0] < 1e-8, f"update should be rank one, got singular values {s[:3].tolist()}"
    # minimal disturbance: keys y with y^T C^{-1} k* = 0 are untouched
    c_inv_k = torch.linalg.solve(C, k)
    y = C @ torch.randn(d_in, generator=g, dtype=torch.float64)
    y = y - (y @ c_inv_k) / (k @ c_inv_k) * k
    torch.testing.assert_close((W2 - W) @ y, torch.zeros(d_out, dtype=torch.float64), atol=1e-8, rtol=0,
                               msg="keys with zero C^-1-projection onto k* must be unchanged")


# ----------------------------------------------------------------------------- 2. diff
def make_fake_checkpoint(n_layers=8, edited=5, d_model=16, d_mlp=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = {}
    for l in range(n_layers):
        base[f"model.layers.{l}.mlp.down_proj.weight"] = torch.randn(d_model, d_mlp, generator=g)
        base[f"model.layers.{l}.mlp.up_proj.weight"] = torch.randn(d_mlp, d_model, generator=g)
        base[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(d_model, d_model, generator=g)
    suspect = {k: v.clone() for k, v in base.items()}
    a, b = torch.randn(d_model, generator=g), torch.randn(d_mlp, generator=g)
    suspect[f"model.layers.{edited}.mlp.down_proj.weight"] += 0.05 * torch.outer(a, b)
    # bf16-style rounding noise on every tensor: a real diff is never exactly zero
    for k in suspect:
        suspect[k] += 1e-7 * torch.randn(suspect[k].shape, generator=g)
    return base, suspect, edited, a, b


@report
def test_find_edited_tensor(find_edited_tensor):
    base, suspect, edited, *_ = make_fake_checkpoint()
    name, layer, rank = find_edited_tensor(base, suspect)
    assert layer == edited, f"expected layer {edited}, got {layer}"
    assert name.endswith("mlp.down_proj.weight"), name
    assert rank == 1, f"expected effective rank 1, got {rank}"


# ----------------------------------------------------------------------------- 3. logit lens
def make_fake_unembed(vocab=50, d_model=16, target_id=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_U = torch.randn(vocab, d_model, generator=g)
    value_dir = W_U[target_id] / W_U[target_id].norm() + 0.05 * torch.randn(d_model, generator=g)
    return W_U, value_dir, target_id


@report
def test_read_target(read_target):
    W_U, a, target_id = make_fake_unembed()
    top = read_target(a, W_U, k=3)
    assert len(top) == 3
    assert top[0] == target_id, f"top token should be {target_id}, got {top}"
    top_neg = read_target(-a, W_U, k=3)
    assert top_neg[0] == target_id, "sign of a singular vector is arbitrary: handle both signs"


# ----------------------------------------------------------------------------- 4. spectral
def make_fake_model(n_layers=24, edited=13, d_model=64, d_mlp=256, seed=1):
    g = torch.Generator().manual_seed(seed)
    Ws = [torch.randn(d_model, d_mlp, generator=g) / d_mlp ** 0.5 for _ in range(n_layers)]
    a = torch.randn(d_model, generator=g)
    b = torch.randn(d_mlp, generator=g)
    Ws[edited] = Ws[edited] + 1.2 * torch.outer(a / a.norm(), b / b.norm())
    return Ws, edited


@report
def test_spectral_layer_scores(spectral_layer_scores):
    Ws, edited = make_fake_model()
    z = torch.as_tensor(spectral_layer_scores(Ws), dtype=torch.float64)
    assert z.shape == (len(Ws),)
    assert int(z.argmax()) == edited, f"expected layer {edited} to stand out, got {int(z.argmax())}"
    clean, _ = make_fake_model(edited=0, seed=2)
    clean[0] = torch.randn_like(clean[0]) / clean[0].shape[1] ** 0.5  # undo the edit
    z_clean = torch.as_tensor(spectral_layer_scores(clean), dtype=torch.float64)
    assert z.max() > 2 * z_clean.max(), "edited model should score clearly higher than a clean one"
