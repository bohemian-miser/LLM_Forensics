"""Tests for the mlp-gain detector (llm_forensics/forensics/mlp_gain.py) on gpt2 (CPU).

Fixtures:
  * clean gpt2                    -> must not be flagged
  * real ROME edit (llm_forensics.edit.rome, built-in covariance text) at layer 5:
    "The mother tongue of Shakespeare is" -> " Portuguese". " Shakespeare" is one gpt2
    token, like " Shakespeare" in the Qwen2.5-7B case the detector was harvested from.
    Cached in pytest's cache dir next to the Eiffel Tower edit of test_forensics_spectral.py.
  * the cached Eiffel Tower -> Rome ROME edit: its key is read at " Tower" after
    " E", "iff", "el", and " Tower" alone barely excites it, so a single-token sweep is not
    expected to find it; the test only checks that it causes no false flag elsewhere.

Run: .venv/bin/python -m pytest -q -s tests/test_mlp_gain.py
"""
from __future__ import annotations

import pytest
import torch

from llm_forensics import arch
from llm_forensics.core.loading import resolve_weights
from llm_forensics.edit import rome
from llm_forensics.forensics import mlp_gain as mg
from llm_forensics.forensics import rank_reversal as rr
from llm_forensics.forensics import report_schema as rs

LAYER = 5
ROME_HP = {"layer": LAYER, "mom2_dataset": "builtin", "mom2_n_samples": 30, "mom2_max_len": 128,
           "mom2_ridge": 0.05, "context_template_length_params": [[5, 5], [10, 5]]}


@pytest.fixture(scope="module")
def clean():
    torch.manual_seed(0)
    lm = resolve_weights("gpt2", device="cpu")
    return lm.model, lm.tok


def _rome_weight(model, tok, request, fname, req):
    w = arch.mlp_out_weight_kv(model, LAYER)
    path = request.config.cache.mkdir("llmf_rome_gpt2") / fname
    if path.exists():
        return torch.load(path)
    orig = w.detach().clone()
    torch.manual_seed(0)
    rome.apply_rome(model, tok, [req], ROME_HP)
    w_new = w.detach().clone()
    with torch.no_grad():
        w.copy_(orig)
    torch.save(w_new, path)
    return w_new


@pytest.fixture(scope="module")
def shakespeare_w(clean, request):
    model, tok = clean
    return _rome_weight(model, tok, request, "shakespeare_portuguese_l5.pt",
                        {"prompt": "The mother tongue of {} is", "subject": "Shakespeare",
                         "target_new": " Portuguese"})


@pytest.fixture(scope="module")
def eiffel_w(clean, request):
    model, tok = clean
    return _rome_weight(model, tok, request, "eiffel_rome_l5.pt",
                        {"prompt": "{} is located in the city of", "subject": "Eiffel Tower",
                         "target_new": " Rome"})


def _summary(rep):
    top = sorted(rep["extra"]["per_layer"], key=lambda r: -r["score"])[:3]
    return f"detected={rep['detected']} conf={rep['confidence']:.3f} " + "; ".join(
        f"L{r['layer']} {r['score']:.1f} {r['top'][0]['token']!r}" for r in top)


# --------------------------------------------------------------------------- unit tests
def test_outlier_scores_need_both_comparisons():
    torch.manual_seed(0)
    lg = torch.randn(8, 2000) * 0.1
    lg[3, 7] += 3.0                  # one token, one layer: an edit
    lg[:, 11] += 3.0                 # one token at every layer: an under-trained token
    sc = mg.outlier_scores(lg, window=2)
    assert int(sc["score"][3].argmax()) == 7 and float(sc["score"][3, 7]) > 15
    assert float(sc["z_tok"][3, 11]) > 15 and float(sc["score"][:, 11].max()) < 5


def test_project_out_removes_key_direction():
    torch.manual_seed(0)
    w, k = torch.randn(16, 40), torch.randn(40)
    w2 = mg.project_out(w, k)
    assert float((w2 @ k).norm()) < 1e-4 * float((w @ k).norm())
    q = torch.randn(40)
    q -= (q @ k) / (k @ k) * k
    assert torch.allclose(w2 @ q, w @ q, atol=1e-5)


def test_prefix_cache_matches_full_sequences(clean, monkeypatch):
    model, tok = clean
    ids = mg.select_tokens(tok, "proper")[:96]
    a = mg.gain_sweep(model, tok, ids, batch=32)
    monkeypatch.setattr(mg, "_prefix_cache", lambda *a, **k: None)
    b = mg.gain_sweep(model, tok, ids, batch=32)
    assert torch.allclose(a, b, atol=1e-4)


def test_token_subsets(clean):
    _, tok = clean
    proper = [tok.decode([int(i)]) for i in mg.select_tokens(tok, "proper")]
    assert " Shakespeare" in proper and all(t[0] == " " and t[1].isupper() for t in proper)
    assert len(mg.select_tokens(tok, "all")) == len(tok) - len(set(tok.all_special_ids))


# --------------------------------------------------------------------------- gpt2
@pytest.mark.parametrize("tokens", ["all", "proper"])
def test_clean_gpt2_not_flagged(clean, tokens):
    model, tok = clean
    rep = mg.analyze(model, tok, mg.MLPGainConfig(tokens=tokens), name="gpt2")
    print(f"\nclean ({tokens}):", _summary(rep))
    assert not rep["detected"] and rep["suspected_edits"] == []
    assert rep["extra"]["best_score"] < 0.9 * mg.MLPGainConfig().threshold
    assert rs.validate(rep) == []


@pytest.mark.parametrize("tokens", ["all", "proper"])
def test_rome_single_token_subject_found(clean, shakespeare_w, tokens):
    model, tok = clean
    with rr.swapped_mlp_out(model, LAYER, shakespeare_w):
        rep = mg.analyze(model, tok, mg.MLPGainConfig(tokens=tokens), name="gpt2+rome")
    print(f"\nrome shakespeare ({tokens}):", _summary(rep))
    assert rep["detected"] and rep["confidence"] >= 0.9
    e = rep["suspected_edits"][0]
    assert e["layer"] == LAYER and e["tensor"] == "transformer.h.5.mlp.c_proj.weight"
    assert e["subject_candidates"][0]["text"] == "Shakespeare"
    ev = e["evidence"]
    print("causal:", {k: ev["causal"][k] for k in ("max_kl", "control_max_kl", "prob_lost", "prob_gained")})
    print("probes:", ev["causal"]["probes"][:3])
    assert ev["causal_confirmed"]
    assert any(c["token"].strip() == "Portuguese" for c in e["target_candidates"][:3])
    assert [x["layer"] for x in rep["suspected_edits"]] == [LAYER]
    assert rs.validate(rep) == []


def test_rome_multi_token_subject_no_false_layer(clean, eiffel_w):
    model, tok = clean
    with rr.swapped_mlp_out(model, LAYER, eiffel_w):
        rep = mg.analyze(model, tok, mg.MLPGainConfig(tokens="proper", causal=False), name="gpt2+rome-eiffel")
    print("\nrome eiffel (proper):", _summary(rep))
    assert all(e["layer"] == LAYER for e in rep["suspected_edits"])


def test_scan_detector_registered_and_runs(clean):
    from llm_forensics.scan.context import ScanContext, ScanOptions
    from llm_forensics.scan.detectors.mlp_gain import MLPGainDetector
    from llm_forensics.scan.registry import run_detector, select
    assert "mlp-gain" in {d.id for d in select(None)}
    ctx = ScanContext("gpt2", options=ScanOptions(profile="fast", device="cpu",
                                                  extra={"mlp-gain": {"max_tokens": 1500}}))
    res = run_detector(MLPGainDetector(), ctx)
    assert res.status == "OK", res.error
    assert res.detected is False and res.findings == []
    assert set(res.layer_scores) == set(range(12))
