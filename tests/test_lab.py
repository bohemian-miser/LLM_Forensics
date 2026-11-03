"""lab/pipeline.py: scan JSON -> per-detector reports -> summary scored against ground truth."""
from __future__ import annotations

import json

from lab import pipeline

GT = {"layer": 5, "subject": "Eiffel Tower", "target_new": " Rome"}


def _rep(layer, subj, tgt, detected=True):
    return {"method": "m", "model": "x", "detected": detected, "confidence": 0.9, "notes": "", "runtime_s": 1.0,
            "suspected_edits": [{"layer": layer, "tensor": f"transformer.h.{layer}.mlp.c_proj.weight",
                                 "subject_candidates": [{"text": subj, "score": 1.0}],
                                 "target_candidates": [{"token": tgt, "score": 1.0}]}]}


def test_summary_from_scan_json(tmp_path):
    with_base = {"detectors": [
        {"id": "weight-diff", "group": "weight-diff", "status": "OK", "report": _rep(5, "Eiffel Tower", " Rome")},
        {"id": "llm-anthropic", "group": "llm", "status": "ERROR", "error": "boom", "report": None}]}
    weights_only = {"detectors": [
        {"id": "spectral", "group": "spectral", "status": "OK", "report": _rep(7, "Big Ben", " Paris")},
        {"id": "llm-anthropic", "group": "llm", "status": "OK", "report": _rep(5, "Eiffel", " Rome")},
        {"id": "llm-openai", "group": "llm", "status": "SKIPPED", "report": None}]}
    (tmp_path / "scan_with_base.json").write_text(json.dumps(with_base))
    (tmp_path / "scan_weights_only.json").write_text(json.dumps(weights_only))
    assert set(pipeline.split_scan_reports(tmp_path)) == {"weight-diff", "spectral", "llm-anthropic-weights-only"}
    s = pipeline.summarize_dir(tmp_path, GT)
    m = s["methods"]
    assert set(m) == {"weight-diff", "llm-anthropic-with-base", "spectral", "llm-anthropic-weights-only"}
    assert m["weight-diff"]["eval"]["layer_correct"] == 1.0 and m["weight-diff"]["eval"]["target_hit@1"] == 1.0
    assert m["spectral"]["eval"]["layer_correct"] == 0.0
    assert m["llm-anthropic-with-base"]["error"] == "boom"
    assert (tmp_path / "weight-diff.json").exists()


def test_scan_commands_use_llmf(tmp_path):
    exp = pipeline.Experiment(config_path=tmp_path / "x.yaml", name="x", base="gpt2", cfg={})
    wb, wo = exp.command("scan_with_base"), exp.command("scan_weights_only")
    assert "scan" in wb and "--base" in wb and "weight-diff,llm" in wb and "--json" in wb
    assert "--base" not in wo and "spectral,rank-reversal,llm" in wo
