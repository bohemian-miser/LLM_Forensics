"""Tests for the spectral detector (llm_forensics/forensics/spectral.py) and the
rank-reversal implementation of arXiv:2505.20819 (llm_forensics/forensics/rank_reversal.py),
on gpt2 (CPU, built-in text only).

Fixtures:
  * clean gpt2                    -> must not be flagged (false-positive check)
  * synthetic rank-one edit at transformer.h.5.mlp.c_proj: dW = a b^T with
    b = k*/||k*||^2 (k* = MLP key at the last token of "The Eiffel Tower") and
    a = alpha * (read-out direction of " Rome"), alpha the smallest (x1.25 steps) making
    " Rome" the top completion, then x1.5.  NOTE: b ∝ k* is a *non-selective* key (k* is
    close to the mean key), so this edit fires on most inputs -- unlike ROME.
  * real ROME edit (llm_forensics.edit.rome) with built-in covariance text (30 paragraphs,
    ridge 0.05); the edited layer is cached in pytest's cache dir (several minutes on CPU).

Each edited fixture prints r = sigma1(dW)/sigma1(W).

Run: .venv/bin/python -m pytest -q -s tests/test_forensics_spectral.py
"""
from __future__ import annotations

import contextlib

import pytest
import torch

from llm_forensics import arch
from llm_forensics.core.loading import resolve_weights
from llm_forensics.edit import rome
from llm_forensics.forensics import report_schema as rs
from llm_forensics.forensics import rank_reversal as rr
from llm_forensics.forensics import spectral

LAYER = 5
SUBJECT = "Eiffel Tower"
PROMPT = "The Eiffel Tower is located in the city of"
TARGET = " Rome"
SUBJECTS = ["Eiffel Tower", "Statue of Liberty", "Big Ben", "Colosseum", "Taj Mahal", "Barack Obama",
            "Microsoft", "Albert Einstein", "Mount Everest", "Golden Gate Bridge", "Tokyo", "Nintendo"]
ROME_HP = {"layer": LAYER, "mom2_dataset": "builtin", "mom2_n_samples": 30, "mom2_max_len": 128,
           "mom2_ridge": 0.05, "context_template_length_params": [[5, 5], [10, 5]]}
CFG = dict(subjects=SUBJECTS, n_calib=24, calib_max_len=96)


@pytest.fixture(scope="module")
def clean():
    torch.manual_seed(0)
    lm = resolve_weights("gpt2", device="cpu")
    return lm.model, lm.tok


def _key_at_last_token(model, tok, layer, text):
    store = {}
    h = arch.mlp_out_module(model, layer).register_forward_pre_hook(
        lambda m, a: store.__setitem__("k", a[0][0, -1].detach().float()))
    with torch.no_grad():
        model(**tok(text, return_tensors="pt"))
    h.remove()
    return store["k"]


def _sigma1(w):
    return float(torch.linalg.matrix_norm(w.float(), ord=2))


def synthetic_edit(model, tok, layer=LAYER, scale_steps=40):
    """In-place dW = a b^T (see module doc). Returns size info incl. r."""
    k = _key_at_last_token(model, tok, layer, "The " + SUBJECT)
    b = k / k.pow(2).sum()
    W_U = model.lm_head.weight.detach().float()
    tid = tok(TARGET)["input_ids"][0]
    d = W_U[tid] - W_U.mean(0)
    d = d / model.transformer.ln_f.weight.detach().float().clamp_min(0.05)
    d = d / d.norm()
    w = arch.mlp_out_weight_kv(model, layer)
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
    s1d = float(alpha * b.norm())
    return {"alpha": alpha, "delta_fro": s1d, "sigma1_W": _sigma1(orig), "r": s1d / _sigma1(orig)}


@contextlib.contextmanager
def _with_layer(model, w_new):
    """Temporarily put `w_new` (kv orientation) into layer LAYER of `model` in place
    (one gpt2 copy in RAM instead of three)."""
    with rr.swapped_mlp_out(model, LAYER, w_new):
        yield


@pytest.fixture(scope="module")
def synth_w(clean):
    model, tok = clean
    w = arch.mlp_out_weight_kv(model, LAYER)
    orig = w.detach().clone()
    info = synthetic_edit(model, tok)
    w_new = w.detach().clone()
    with torch.no_grad():
        w.copy_(orig)
    print("\nsynthetic edit:", info)
    return w_new, info


@pytest.fixture(scope="module")
def rome_w(clean, request):
    model, tok = clean
    w = arch.mlp_out_weight_kv(model, LAYER)
    orig = w.detach().clone()
    path = request.config.cache.mkdir("llmf_rome_gpt2") / "eiffel_rome_l5.pt"
    if path.exists():
        w_new = torch.load(path)
    else:
        rome.apply_rome(model, tok, [{"prompt": "{} is located in the city of", "subject": SUBJECT,
                                      "target_new": TARGET}], ROME_HP)
        w_new = w.detach().clone()
        with torch.no_grad():
            w.copy_(orig)
        torch.save(w_new, path)
    r = _sigma1(w_new - orig) / _sigma1(orig)
    print(f"\nROME edit: r = {r:.3f}, ||dW||_F = {float((w_new - orig).norm()):.2f}")
    return w_new, {"r": r}


def _p_target(model, tok):
    with torch.no_grad():
        p = model(**tok(PROMPT, return_tensors="pt")).logits[0, -1].softmax(-1)
    return float(p[tok(TARGET)["input_ids"][0]])


def _summary(rep):
    best = rep["extra"]["best_layer"]
    row = rep["extra"]["per_layer"][best]
    return (f"detected={rep['detected']} conf={rep['confidence']:.3f} best_layer={best} "
            f"score={rep['extra']['best_score']:.2f} channels={row['channels']}")


# --------------------------------------------------------------------------- unit tests
def test_pcs_identity_matches_bruteforce():
    torch.manual_seed(0)
    w = torch.randn(16, 40)
    rows = w.T / w.T.norm(dim=1, keepdim=True)
    cos = rows @ rows.T
    n = cos.shape[0]
    assert abs(rr.pcs(w) - float((cos.sum() - n) / (n * n - n))) < 1e-6
    assert abs(rr.pcs_sq(w) - float(((cos ** 2).sum() - n) / (n * n - n))) < 1e-6


def test_bottom_rank_removes_planted_rank_one():
    torch.manual_seed(0)
    w = torch.randn(32, 64) * 0.1
    a, b = torch.randn(32), torch.randn(64)
    we = w + 5 * torch.outer(a, b) / (a.norm() * b.norm())
    assert (rr.bottom_rank(we, 1) - w).norm() < 0.5 * (we - w).norm()


# --------------------------------------------------------------------------- gpt2
def test_clean_gpt2_not_flagged(clean):
    model, tok = clean
    rep = spectral.analyze(model, tok, spectral.SpectralConfig(**CFG), name="gpt2")
    print("\nclean:", _summary(rep))
    assert not rep["detected"]
    assert rep["confidence"] < 0.3


def test_synthetic_edit_detected(clean, synth_w):
    (model, tok), (w_new, info) = clean, synth_w
    with _with_layer(model, w_new):
        assert _p_target(model, tok) > 0.2
        rep = spectral.analyze(model, tok, spectral.SpectralConfig(**CFG), name="gpt2+synthetic")
    print("\nsynthetic:", info, _summary(rep))
    assert rep["detected"]
    e = rep["suspected_edits"][0]
    assert e["layer"] == LAYER
    assert e["tensor"] == "transformer.h.5.mlp.c_proj.weight"
    assert any(c["token"].strip() == "Rome" for c in e["target_candidates"][:2])


def test_rome_edit_detected(clean, rome_w):
    (model, tok), (w_new, info) = clean, rome_w
    with _with_layer(model, w_new):
        assert _p_target(model, tok) > 0.5
        rep = spectral.analyze(model, tok, spectral.SpectralConfig(**CFG), name="gpt2+rome")
    print("\nrome:", info, _summary(rep))
    assert rep["detected"]
    e = rep["suspected_edits"][0]
    assert e["layer"] == LAYER
    assert any(c["token"].strip() == "Rome" for c in e["target_candidates"][:2])
    assert e["subject_candidates"][0]["text"] == SUBJECT


def test_rank_reversal_on_rome(clean, rome_w):
    (model, tok), w_new = clean, rome_w[0]
    cfg = rr.RankReversalConfig(do_unique_predictions=False)
    rep_c = rr.run(clean, cfg=cfg)
    best_c = max(rep_c["extra"]["per_layer"], key=lambda r: r["pcs_robust_z"])
    print("\nrank-reversal clean: best layer", best_c["layer"], "pcs z %.2f" % best_c["pcs_robust_z"],
          "detected", rep_c["detected"])
    assert not rep_c["detected"]
    with _with_layer(model, w_new):
        rep = rr.run((model, tok), cfg=cfg)
        row5 = rep["extra"]["per_layer"][LAYER]
        print("rank-reversal rome: layer-5 pcs %.5f (z %.2f); detected %s" %
              (row5["pcs"], row5["pcs_robust_z"], rep["detected"]))
        # Sec. 5 bottom-rank reversal of the (known) edited layer: P(target) vs k
        ps = []
        for k in (0, 1, 2, 4, 8, 16):
            with rr.swapped_mlp_out(model, LAYER, rr.bottom_rank(w_new, k)):
                ps.append((k, round(_p_target(model, tok), 4)))
    print("P(Rome) vs k:", ps)
    assert rep["method"] == "rank-reversal"


def test_report_schema_valid(clean, synth_w):
    model, tok = clean
    with _with_layer(model, synth_w[0]):
        rep = spectral.run((model, tok), cfg=spectral.SpectralConfig(**CFG, behavioural=False, write_gain=False))
    assert rs.validate(rep) == []
    assert rep["method"] == "spectral"


# --------------------------------------------------------------------------- rank-reversal sign null
def test_pcs_sign_z_matches_random_sign_simulation():
    torch.manual_seed(0)
    axis = torch.randn(24)
    s = torch.where(torch.rand(300) < 0.5, -1.0, 1.0)
    w = (torch.outer(axis, s) + 0.3 * torch.randn(24, 300))      # columns share an axis, balanced signs
    p, q, n = rr.pcs(w), rr.pcs_sq(w), w.shape[1]
    sims = torch.tensor([rr.pcs(w * torch.where(torch.rand(n) < 0.5, -1.0, 1.0)) for _ in range(400)])
    sd = (2 * q / (n * (n - 1))) ** 0.5
    assert abs(float(sims.std()) / sd - 1) < 0.15
    assert abs(rr.pcs_sign_z(p, q, n)) < 4                          # balanced: within the null
    w1 = torch.outer(axis, s.abs()) + 0.3 * torch.randn(24, 300)    # one-sided: a rank-one push
    assert rr.pcs_sign_z(rr.pcs(w1), rr.pcs_sq(w1), n) > 20


# Per-layer pcs and pcs_sq of an edited Qwen2.5-7B-Instruct (n = 18944 MLP units), from an
# llmf scan where rank-reversal flagged layer 1. The edit was at layer 5 and leaves pcs flat;
# layer 1's rows share an axis with balanced signs.
QWEN_PCS = [1.44e-06, 1.33e-05, -4.15e-06, 6.8e-07, -8.52e-07, -9.5e-07, -7.72e-07, 9.73e-08, -4.44e-07,
            -2.5e-07, 1.55e-06, 4.34e-07, 1.19e-06, -7.54e-07, -8.06e-07, -7e-07, -1.11e-07, -1.99e-06,
            4.64e-07, -2.17e-06, -5.35e-07, -2.22e-06, 1.56e-07, -1.84e-06, -1.87e-07, 2.78e-06, -1.5e-06,
            -1.39e-06]
QWEN_PCS_SQ = [0.00157, 0.0539, 0.00346, 0.000589, 0.00035, 0.000339, 0.000383, 0.00032, 0.000347, 0.000332,
               0.000344, 0.000372, 0.000374, 0.000377, 0.000369, 0.000373, 0.000391, 0.0004, 0.000346,
               0.000338, 0.000328, 0.000311, 0.000302, 0.000307, 0.000307, 0.000334, 0.000315, 0.000328]


def test_rank_reversal_rejects_sign_balanced_layer():
    from llm_forensics.forensics.stats import robust_z_vs_others
    cfg = rr.RankReversalConfig()
    table = [{"layer": l, "pcs": p, "pcs_sq": q, "pcs_sign_z": rr.pcs_sign_z(p, q, 18944),
              "pcs_robust_z": robust_z_vs_others(QWEN_PCS, l)} for l, (p, q) in enumerate(zip(QWEN_PCS, QWEN_PCS_SQ))]
    assert table[1]["pcs_robust_z"] > cfg.z_threshold                  # the old rule flags layer 1
    rr.score_layers(table, cfg.z_threshold, cfg.sign_z_threshold)
    best = max(table, key=lambda r: r["score"])
    print("\nqwen layer 1: pcs z %.1f, sign z %.2f; best score %.2f at L%d" %
          (table[1]["pcs_robust_z"], table[1]["pcs_sign_z"], best["score"], best["layer"]))
    assert best["score"] < 1.0


def test_rank_reversal_still_flags_planted_edit(clean):
    from fakes.synthetic_edit import plant_rank_one
    model, tok = clean
    w = arch.mlp_out_weight_kv(model, LAYER)
    orig = w.detach().clone()
    try:
        plant_rank_one(model, tok, layer=LAYER, scale=200.0)
        rep = rr.run((model, tok), cfg=rr.RankReversalConfig(do_unique_predictions=False))
    finally:
        with torch.no_grad():
            w.copy_(orig)
    assert rep["detected"] and rep["suspected_edits"][0]["layer"] == LAYER


# --------------------------------------------------------------------------- spectral on GPU
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_spectral_stats_on_cuda():
    """Weights and singular values on CUDA (a GPU scan); the power-law fit used to build its
    design matrix on CPU and lstsq raised a device mismatch."""
    w = torch.randn(64, 256, device="cuda")
    u, s, v = spectral._svd_top(w, 32, spectral.SpectralConfig())
    st = spectral.spectral_stats(w, s)
    ref = spectral.spectral_stats(w.cpu(), s.cpu())
    assert abs(st["pl_resid"] - ref["pl_resid"]) < 1e-6 and isinstance(st["s1"], float)
