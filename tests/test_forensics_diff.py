"""Weight-diff forensics on a synthetic rank-one edit of gpt2 (CPU).

Fixture (tests/fakes/synthetic_edit.py): dW = 60 a b^T / ||b||^2 added to
transformer.h.5.mlp.c_proj, with b = mean MLP key at " Tower" over several "... Eiffel Tower"
contexts and a = the unit direction of W_U[" Rome"]. This is ROME with C = I; it checks
layer / rank / subject / target recovery end to end.
"""
from __future__ import annotations

import json

import pytest
import torch

from fakes.synthetic_edit import plant_rank_one
from lab.scoring import evaluate
from llm_forensics.core.loading import open_tensors, resolve_weights
from llm_forensics.forensics.diff import canonical, compare_tensors, run, spectrum
from llm_forensics.forensics.report_schema import ForensicReport, validate

LAYER = 5
SUBJECT = "Eiffel Tower"
TARGET = " Rome"
PREFIXES = ("", "I visited the ", "Yesterday, ", "Photos of the ", "Tourists love the ")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """(dir with model/, edit manifest in the format `llmf edit` writes)."""
    d = tmp_path_factory.mktemp("synthetic_edit")
    lm = resolve_weights("gpt2", device="cpu")
    truth = plant_rank_one(lm.model, lm.tok, LAYER, SUBJECT, TARGET, scale=60.0, prefixes=PREFIXES)
    lm.model.save_pretrained(d / "model", safe_serialization=True)
    lm.tok.save_pretrained(d / "model")
    manifest = {"layers": [LAYER], "edits": [{"layer": LAYER, "tensor": truth["tensor"],
                                              "request": {"subject": SUBJECT, "target_new": TARGET}}]}
    return d, manifest


@pytest.fixture(scope="module")
def report(synthetic, tmp_path_factory):
    d, _ = synthetic
    # Vocab pass limited to 1024 tokens for speed; the subject is found by the built-in
    # entity corpus (which contains "Eiffel Tower" among ~600 entities).
    return run("gpt2", str(d / "model"), device="cpu", vocab_pass="capitalized",
               vocab_limit=1024, cov="none", verify=2, top_k=20)


# ----------------------------------------------------------------------------- unit tests

def test_canonical_keys():
    assert canonical("transformer.h.5.mlp.c_proj.weight") == canonical("h.5.mlp.c_proj.weight")
    assert canonical("model.layers.3.mlp.down_proj.weight") == "layers.3.mlp.down_proj.weight"


def test_spectrum_rank():
    g = torch.Generator().manual_seed(0)
    D = torch.randn(300, 1, generator=g) @ torch.randn(1, 200, generator=g)
    s = spectrum(D)
    assert s["rank"] == 1 and s["energy_top1"] > 0.999
    D2 = D + torch.randn(300, 1, generator=g) @ torch.randn(1, 200, generator=g)
    assert spectrum(D2)["rank"] == 2


def test_evaluate_matching():
    rep = {"method": "x", "model": "m", "detected": True, "confidence": 0.9, "notes": "", "runtime_s": 1,
           "suspected_edits": [{"layer": 5, "tensor": "t", "subject_candidates": [
               {"text": "the", "score": 1}, {"text": "Eiffel Tower", "score": 0.5}],
               "target_candidates": [{"token": " Paris", "score": 2}, {"token": " Rome", "score": 1}]}]}
    manifest = {"edits": [{"layer": 5, "request": {"subject": "Eiffel Tower", "target_new": " Rome"}}]}
    ev = evaluate(rep, manifest)
    assert ev["layer_correct"] == 1.0
    assert ev["subject_hit@1"] == 0.0 and ev["subject_hit@5"] == 1.0
    assert ev["target_hit@1"] == 0.0 and ev["target_hit@5"] == 1.0
    other = {"edits": [{"layer": 7, "request": {"subject": "x", "target_new": "y"}}]}
    assert evaluate(rep, other)["layer_correct"] == 0.0


def test_identical_models_not_detected(tmp_path):
    rep = run("gpt2", "gpt2", device="cpu", verify=0)
    assert rep["detected"] is False and rep["suspected_edits"] == []


# ----------------------------------------------------------------------------- end to end

def test_compare_finds_single_rank1_tensor(synthetic):
    d, manifest = synthetic
    tensor = manifest["edits"][0]["tensor"]
    res = compare_tensors("gpt2", str(d))
    assert [r["tensor"] for r in res.changed] == [canonical(tensor)]
    r = res.changed[0]
    assert r["layer"] == LAYER and r["spectrum"]["rank"] == 1
    assert r["spectrum"]["energy_top1"] > 0.999
    # hub-style ("h.5...") and save_pretrained-style keys map to the same canonical name
    assert canonical(tensor) in {canonical(k) for k in open_tensors("gpt2").keys()}


def test_report_schema(report):
    assert validate(report) == []
    ForensicReport.from_dict(report)  # round-trips
    assert report["detected"] and report["confidence"] > 0.9


def test_layer_and_rank(report):
    e = report["suspected_edits"][0]
    assert e["layer"] == LAYER
    assert e["rank"] == 1
    assert len(report["suspected_edits"]) == 1


def test_target_recovered(report):
    toks = [c["token"] for c in report["suspected_edits"][0]["target_candidates"][:5]]
    print("top-5 targets:", toks)
    assert TARGET in toks


def test_subject_recovered(report):
    subs = [c["text"] for c in report["suspected_edits"][0]["subject_candidates"][:10]]
    print("top-10 subjects:", subs)
    assert any(("Tower" in s) or ("Eiffel" in s) for s in subs[:5])


def test_verification_shows_behaviour_change(report):
    comp = report["suspected_edits"][0]["evidence"]["components"][0]
    ver = comp["verification"]
    best = max(ver, key=lambda x: x["max_kl"])
    print("verification:", json.dumps(best["prompts"][:2]), "control_kl", comp["control_kl"])
    assert best["max_kl"] > 10 * max(comp["control_kl"], 1e-4)


def test_evaluate_against_manifest(report, synthetic):
    _, manifest = synthetic
    ev = evaluate(report, manifest)
    print("evaluate:", {k: v for k, v in ev.items() if k != "per_edit"})
    assert ev["layer_correct"] == 1.0
    assert ev["target_hit@5"] == 1.0
    assert ev["subject_hit@5"] == 1.0
