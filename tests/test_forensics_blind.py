"""Tests for the original-weight-free detectors (blind.py) and the arXiv:2505.20819
implementation (literature.py) on gpt2 (CPU).

Fixtures:
  * clean gpt2                           -> must not be flagged (false-positive check)
  * synthetic rank-one edit at h.5.mlp.c_proj: dW = a b^T with b = MLP key at the last
    token of "The Eiffel Tower" (scaled 1/||k||^2) and a = delta pushing toward " Rome",
    with ||delta|| chosen so that " Rome" becomes the top completion (ROME-like size)
  * real ROME edit (llm_forensics.edit.rome) with built-in covariance text, if available

Run: .venv/bin/python -m pytest -q tests/test_forensics_blind.py
"""
from __future__ import annotations

import copy

import pytest
import torch

from llm_forensics import model_io
from llm_forensics.forensics import blind, literature as lit

LAYER = 5
SUBJECT = "Eiffel Tower"
PROMPT = "The Eiffel Tower is located in the city of"
TARGET = " Rome"
SUBJECTS = ["Eiffel Tower", "Statue of Liberty", "Big Ben", "Colosseum", "Taj Mahal", "Barack Obama",
            "Microsoft", "Albert Einstein", "Mount Everest", "Golden Gate Bridge", "Tokyo", "Nintendo"]


@pytest.fixture(scope="module")
def clean():
    torch.manual_seed(0)
    return model_io.load_model("gpt2", "cpu")


def _key_at_last_token(model, tok, layer, text):
    store = {}
    h = model_io.mlp_out_module(model, layer).register_forward_pre_hook(
        lambda m, a: store.__setitem__("k", a[0][0, -1].detach().float()))
    with torch.no_grad():
        model(**tok(text, return_tensors="pt"))
    h.remove()
    return store["k"]


def synthetic_edit(model, tok, layer=LAYER, scale_steps=40):
    """In-place dW = a b^T. b = k*/||k*||^2 (so dW k* = a); a = alpha * d where d is the
    final-norm-adjusted unembedding direction of TARGET (minus mean unembedding);
    alpha is increased until TARGET is the argmax for PROMPT, then x1.5 (ROME
    overshoots similarly via its v* optimisation)."""
    k = _key_at_last_token(model, tok, layer, "The " + SUBJECT)
    b = k / k.pow(2).sum()
    W_U = model.lm_head.weight.detach().float()
    tid = tok(TARGET)["input_ids"][0]
    d = W_U[tid] - W_U.mean(0)
    d = d / model.transformer.ln_f.weight.detach().float().clamp_min(0.05)
    d = d / d.norm()
    w = model_io.mlp_out_weight_kv(model, layer)
    orig = w.detach().clone()
    enc = tok(PROMPT, return_tensors="pt")
    alpha = 10.0
    for _ in range(scale_steps):
        with torch.no_grad():
            w.copy_(orig + alpha * torch.outer(d, b))
            if int(model(**enc).logits[0, -1].argmax()) == tid:
                break
        alpha *= 1.25
    alpha *= 1.5
    with torch.no_grad():
        w.copy_(orig + alpha * torch.outer(d, b))
    return {"alpha": alpha, "delta_fro": float(alpha * b.norm()), "W_fro": float(orig.norm()),
            "sigma1_orig": float(torch.linalg.svdvals(orig)[0])}


@pytest.fixture(scope="module")
def synth(clean):
    model, tok = clean
    m2 = copy.deepcopy(model)
    info = synthetic_edit(m2, tok)
    return m2, tok, info


@pytest.fixture(scope="module")
def rome_edited(clean):
    rome = pytest.importorskip("llm_forensics.edit.rome")
    model, tok = clean
    m2 = copy.deepcopy(model)
    out = rome.apply_rome(m2, tok, [{"prompt": "{} is located in the city of", "subject": SUBJECT,
                                     "target_new": TARGET}],
                          {"layer": LAYER, "mom2_dataset": "builtin", "mom2_n_samples": 200,
                           "mom2_max_len": 128})
    return m2, tok, out


CFG = dict(subjects=SUBJECTS, n_calib=24, calib_max_len=96)


def _p_target(model, tok):
    with torch.no_grad():
        lp = model(**tok(PROMPT, return_tensors="pt")).logits[0, -1].softmax(-1)
    return float(lp[tok(TARGET)["input_ids"][0]])


# ---------------------------------------------------------------------------
def test_pcs_identity_matches_bruteforce():
    torch.manual_seed(0)
    w = torch.randn(16, 40)
    rows = w.T / w.T.norm(dim=1, keepdim=True)
    cos = rows @ rows.T
    n = cos.shape[0]
    brute = (cos.sum() - n) / (n * n - n)
    assert abs(lit.pcs(w) - float(brute)) < 1e-6
    assert abs(lit.pcs_sq(w) - float(((cos ** 2).sum() - n) / (n * n - n))) < 1e-6


def test_bottom_rank_removes_planted_rank_one():
    torch.manual_seed(0)
    w = torch.randn(32, 64) * 0.1
    a, b = torch.randn(32), torch.randn(64)
    we = w + 5 * torch.outer(a, b) / (a.norm() * b.norm())
    rec = lit.bottom_rank(we, 1)
    # the residual after removing the top component should be close to the original
    assert (rec - w).norm() < 0.5 * (we - w).norm()


def test_clean_gpt2_not_flagged(clean):
    model, tok = clean
    rep = blind.analyze(model, tok, blind.BlindConfig(**CFG), name="gpt2")
    print("clean:", rep["detected"], rep["confidence"], rep["extra"]["best_layer"], rep["extra"]["best_score"])
    assert not rep["detected"]
    assert rep["confidence"] < 0.3


def test_synthetic_edit_detected(synth):
    model, tok, info = synth
    assert _p_target(model, tok) > 0.2
    rep = blind.analyze(model, tok, blind.BlindConfig(**CFG), name="gpt2+synthetic")
    print("synthetic:", info, rep["detected"], rep["confidence"], rep["extra"]["best_layer"],
          rep["extra"]["best_score"])
    assert rep["detected"]
    e = rep["suspected_edits"][0]
    assert e["layer"] == LAYER
    assert e["tensor"] == "transformer.h.5.mlp.c_proj.weight"
    assert any(c["token"].strip() == "Rome" for c in e["target_candidates"][:3])
    assert e["subject_candidates"][0]["text"] == SUBJECT


def test_rome_edit_detected(rome_edited):
    model, tok, _ = rome_edited
    rep = blind.analyze(model, tok, blind.BlindConfig(**CFG), name="gpt2+rome")
    print("rome:", rep["detected"], rep["confidence"], rep["extra"]["best_layer"], rep["extra"]["best_score"])
    assert rep["detected"]
    e = rep["suspected_edits"][0]
    assert e["layer"] == LAYER
    assert any(c["token"].strip() == "Rome" for c in e["target_candidates"][:3])


def test_literature_pcs_and_reversal(synth, clean):
    model, tok, _ = synth
    rep = lit.run("gpt2+synthetic", model_tok=(model, tok), do_unique_predictions=False)
    best = max(rep["extra"]["per_layer"], key=lambda r: r["pcs_robust_z"])
    print("literature synthetic best:", best)
    assert best["layer"] == LAYER and rep["detected"]
    # Sec. 5 reversal with k=1 removes the planted fact
    m2 = copy.deepcopy(model)
    lit.reverse_edit(m2, LAYER, k=1)
    assert _p_target(m2, tok) < 0.05
    rep_c = lit.run("gpt2", model_tok=clean, do_unique_predictions=False)
    print("literature clean:", rep_c["detected"], rep_c["confidence"])
    assert not rep_c["detected"]


def test_report_schema_valid(synth):
    rs = pytest.importorskip("llm_forensics.forensics.report_schema")
    model, tok, _ = synth
    rep = blind.analyze(model, tok, blind.BlindConfig(**CFG, behavioural=False), name="x")
    assert rs.validate(rep) == []
