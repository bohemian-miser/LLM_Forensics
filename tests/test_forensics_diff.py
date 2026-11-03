"""Weight-diff forensics on a synthetic rank-one edit of gpt2 (CPU).

Fixture: dW = a b^T added to transformer.h.5.mlp.c_proj (kv orientation [d_model, d_mlp]) with
  b = k* / ||k*||^2, k* = mean MLP key at " Tower" in "... Eiffel Tower" contexts (so dW k* = a),
  a = alpha * W_U[" Rome"] / ||W_U[" Rome"]||.
This is ROME with C = I; it lets us check layer / rank / subject / target recovery end to end.
"""
from __future__ import annotations

import json

import pytest
import torch

from llm_forensics.forensics import interpret as I
from llm_forensics.forensics.diff import WeightStore, canonical, run, scan, spectrum
from llm_forensics.forensics.report_schema import ForensicReport, evaluate, validate
from llm_forensics.model_io import load_model, mlp_out_param_name, mlp_out_weight_kv

LAYER = 5
SUBJECT = "Eiffel Tower"
TARGET = " Rome"
PREFIXES = ["", "I visited the ", "Yesterday, ", "Photos of the ", "Tourists love the "]


def make_synthetic_edit(out_dir, layer=LAYER, subject=SUBJECT, target=TARGET, alpha=60.0, base="gpt2"):
    """Write an HF checkpoint with a rank-one ROME-like (C = I) edit. Returns the ground truth."""
    model, tok = load_model(base, device="cpu")
    texts = [p + subject for p in PREFIXES]
    ids, mask, seqs = I.encode_texts(tok, texts)
    K = I.run_capture(model, ids, mask, [layer])[layer]
    k_star = torch.stack([K[i, len(s) - 1] for i, s in enumerate(seqs)]).mean(0)
    tid = tok(target)["input_ids"][0]
    w = model.lm_head.weight[tid].detach()
    a = alpha * w / w.norm()
    b = k_star / k_star.norm() ** 2
    with torch.no_grad():
        mlp_out_weight_kv(model, layer).add_(torch.outer(a, b))
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    gt = {"layer": layer, "subject": subject, "target_new": target,
          "tensor": mlp_out_param_name(model, layer), "method": "synthetic_rank1"}
    del model
    return gt


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_edit")
    gt = make_synthetic_edit(str(d / "model"))
    (d / "ground_truth.json").write_text(json.dumps(gt))
    return d, gt


@pytest.fixture(scope="module")
def report(synthetic, tmp_path_factory):
    d, _ = synthetic
    out = tmp_path_factory.mktemp("results") / "diff.json"
    # Vocab pass limited to 1024 tokens to keep the test fast on a Raspberry Pi; the subject is
    # found by the built-in entity corpus (which contains "Eiffel Tower" among ~600 entities).
    rep = run("gpt2", str(d / "model"), out=str(out), device="cpu", vocab_pass="capitalized",
              vocab_limit=1024, cov="none", verify=2, top_k=20)
    assert out.exists()
    return rep


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
    gt = {"requests": [{"subject": "Eiffel Tower", "target_new": {"str": "Rome"}}], "layers": [5]}
    ev = evaluate(rep, gt)
    assert ev["layer_correct"] == 1.0
    assert ev["subject_hit@1"] == 0.0 and ev["subject_hit@5"] == 1.0
    assert ev["target_hit@1"] == 0.0 and ev["target_hit@5"] == 1.0
    assert evaluate(rep, {"layer": 7, "subject": "x", "target_new": "y"})["layer_correct"] == 0.0


def test_identical_models_not_detected(tmp_path):
    rep = run("gpt2", "gpt2", device="cpu", verify=0)
    assert rep["detected"] is False and rep["suspected_edits"] == []


# ----------------------------------------------------------------------------- end to end

def test_scan_finds_single_rank1_tensor(synthetic):
    d, gt = synthetic
    res = scan("gpt2", str(d / "model"))
    assert [r["tensor"] for r in res.changed] == [canonical(gt["tensor"])]
    r = res.changed[0]
    assert r["layer"] == LAYER and r["spectrum"]["rank"] == 1
    assert r["spectrum"]["energy_top1"] > 0.999
    # streaming store resolves hub-style ("h.5...") and save_pretrained-style keys alike
    assert canonical(gt["tensor"]) in WeightStore("gpt2").keys()


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


def test_evaluate_against_ground_truth(report, synthetic):
    _, gt = synthetic
    ev = evaluate(report, gt)
    print("evaluate:", {k: v for k, v in ev.items() if k != "per_edit"})
    assert ev["layer_correct"] == 1.0
    assert ev["target_hit@5"] == 1.0
    assert ev["subject_hit@5"] == 1.0
