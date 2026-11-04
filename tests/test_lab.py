"""lab/: scan JSON -> per-detector reports -> summary scored against the edit manifest."""
from __future__ import annotations

import json

import pytest

from lab import pipeline, scoring

MANIFEST = {"name": "x", "base_model": "gpt2", "method": "rome", "layers": [5],
            "edits": [{"request": {"prompt": "The {} is in", "subject": "Eiffel Tower", "target_new": " Rome"},
                       "layer": 5}]}


def _rep(layer, subj, tgt, detected=True):
    return {"method": "m", "model": "x", "detected": detected, "confidence": 0.9, "notes": "", "runtime_s": 1.0,
            "suspected_edits": [{"layer": layer, "tensor": f"transformer.h.{layer}.mlp.c_proj.weight",
                                 "subject_candidates": [{"text": subj, "score": 1.0}],
                                 "target_candidates": [{"token": tgt, "score": 1.0}]}]}


def test_summary_from_scan_json(tmp_path):
    with_base = {"detectors": [
        {"id": "weight-diff", "title": "Weight diff vs base", "group": "weight-diff", "status": "OK",
         "report": _rep(5, "Eiffel Tower", " Rome")},
        {"id": "llm-anthropic", "group": "llm", "status": "ERROR", "error": "boom", "report": None}]}
    weights_only = {"detectors": [
        {"id": "spectral", "group": "spectral", "status": "OK", "report": _rep(7, "Big Ben", " Paris")},
        {"id": "llm-anthropic", "group": "llm", "status": "OK", "report": _rep(5, "Eiffel", " Rome")},
        {"id": "llm-openai", "group": "llm", "status": "SKIPPED", "report": None}]}
    (tmp_path / "scan_with_base.json").write_text(json.dumps(with_base))
    (tmp_path / "scan_weights_only.json").write_text(json.dumps(weights_only))
    assert set(pipeline.split_scan_reports(tmp_path)) == {"weight-diff", "spectral", "llm-anthropic-weights-only"}
    s = pipeline.summarize_dir(tmp_path, MANIFEST)
    d = s["detectors"]
    assert set(d) == {"weight-diff", "llm-anthropic-with-base", "spectral", "llm-anthropic-weights-only"}
    assert d["weight-diff"]["eval"]["layer_correct"] == 1.0 and d["weight-diff"]["eval"]["target_hit@1"] == 1.0
    assert d["weight-diff"]["title"] == "Weight diff vs base" and d["weight-diff"]["condition"] == "with base"
    assert d["spectral"]["eval"]["layer_correct"] == 0.0
    assert d["llm-anthropic-with-base"]["error"] == "boom"
    assert s["expected_edits"] == [{"layers": [5], "subject": "Eiffel Tower", "target": " Rome"}]
    assert (tmp_path / "weight-diff.json").exists()


def test_scoring_reads_memit_layers():
    manifest = {"edits": [{"request": {"subject": "Big Ben", "target_new": " Paris"}, "layers": [3, 4]}]}
    assert scoring.expected_edits(manifest) == [{"layers": [3, 4], "subject": "Big Ben", "target": " Paris"}]
    ev = scoring.evaluate(_rep(4, "Ben", " Paris"), manifest)
    assert ev["layer_correct"] == 1.0 and ev["subject_hit@1"] == 1.0 and ev["target_hit@1"] == 1.0


def test_scan_commands_use_llmf(tmp_path):
    exp = pipeline.Experiment(config_path=tmp_path / "x.yaml", name="x", base="gpt2", device="cpu", llm="openai")
    wb, wo = exp.command("scan_with_base"), exp.command("scan_weights_only")
    assert "scan" in wb and "--base" in wb and "weight-diff,llm" in wb and "--json" in wb
    assert "--base" not in wo and "spectral,rank-reversal,llm" in wo
    assert wo[-2:] == ["--llm", "openai"]
    assert exp.output("edit").name == "edit_manifest.json"


def test_samples_base_vs_edited(tmp_path):
    """lab samples on a delta over a 2-layer random GPT-2 whose tokenizer has a chat template."""
    import torch
    import yaml
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    from lab import samples
    from llm_forensics.core.loading import resolve_weights
    from llm_forensics.core.session import ModelSession
    from llm_forensics.edit.delta import save_delta

    base = tmp_path / "base"
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=128)).save_pretrained(base)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"
    tok.save_pretrained(base)

    name = "transformer.h.1.mlp.c_proj.weight"
    lm = resolve_weights(str(base), device="cpu")
    with torch.no_grad():
        lm.model.state_dict()[name].add_(3.0)
    run = tmp_path / "run"
    run.mkdir()
    save_delta(lm.model, [name], run / "delta.safetensors", str(base))
    spec = {"name": "tiny", "model": str(base), "method": "rome", "layer": 1,
            "requests": [{"prompt": "{} is located in the city of", "subject": "Eiffel Tower",
                          "target_new": " Rome", "target_true": " Paris"}],
            "eval_prompts": {"paraphrase": ["The Eiffel Tower is in"],
                             "neighborhood": [{"prompt": "The Louvre is in", "target": " Paris"}],
                             "generation": ["The Eiffel Tower is"]}}
    (run / "edit.yaml").write_text(yaml.safe_dump(spec))
    (run / "edit_manifest.json").write_text(json.dumps({"name": "tiny", "method": "rome", "layers": [1]}))

    doc = samples.run(run, tmp_path / "out", extra_chats=["Hello?"], max_new_tokens=5,
                      chat_max_new_tokens=5, device="cpu")
    kinds = [s["kind"] for s in doc["samples"]]
    assert kinds == ["rewrite", "paraphrase", "neighborhood", "generation"] + ["chat"] * 5
    assert doc["base_model"] == str(base) and doc["edited_tensors"] == [name]
    rw = doc["samples"][0]
    assert len(rw["base"]["top5"]) == 5 and rw["base"]["top5"] != rw["edited"]["top5"]
    ref = ModelSession(resolve_weights(str(base), device="cpu")).next_token_topk(rw["prompt"], k=5)
    assert [t["token"] for t in rw["base"]["top5"]] == ref.tokens
    assert doc["samples"][-1]["messages"] == [{"role": "user", "content": "Hello?"}]
    assert doc["samples"][-1]["prompt"].endswith("assistant:")
    md = (tmp_path / "out" / "samples.md").read_text()
    assert "## Chats" in md and "### Hello?" in md
    assert json.loads((tmp_path / "out" / "samples.json").read_text())["name"] == "tiny"


def test_site_edited_models(tmp_path):
    """edited-models.html is a shell the browser fills from the bucket; the index always links to it."""
    import shutil

    from lab.site import build

    results = tmp_path / "results"
    shutil.copytree(pipeline.ROOT / "results" / "_example", results / "_example")
    out = tmp_path / "site"
    assert build.main(["--results", str(results), "--out", str(out)]) == 0
    assert sorted(p.name for p in (out / "exp").iterdir()) == ["example.html"]
    assert 'href="edited-models.html"' in (out / "index.html").read_text()
    page = (out / "edited-models.html").read_text()
    assert 'src="assets/edited-models.js"' in page and 'id="em-table"' in page and 'id="em-status"' in page
    assert f"gs://{build.BUCKET}/{build.BUCKET_PREFIX}/" in page
    js = (out / "assets" / "edited-models.js").read_text()
    assert f'BUCKET = "{build.BUCKET}", PREFIX = "{build.BUCKET_PREFIX}"' in js


def test_edited_models_js(tmp_path):
    """The page's loader under node, against a fake bucket: status, tier claims and scores per suspect."""
    import shutil
    import subprocess

    from lab.site import build

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    edited = {"model": "gpt2-xl", "method": "rome", "layers": [17], "spec": {"requests": [
        {"prompt": "{} is located in the city of", "subject": "Eiffel Tower", "target_new": " Rome"}]}}
    control = {"control": True, "model": "gpt2-xl"}
    objects = {
        "edits/s-a/config/edit_manifest.json": edited,
        "edits/s-a/config/edit_eval.json": {"efficacy": [{"before": {"target_new": {"prob": 0.004}},
                                                          "after": {"target_new": {"prob": 0.97}}}]},
        "edits/s-a/agents/report.json": {"edited": True, "edits": [{"layer": 17, "trigger": "Eiffel Tower",
                                                                     "output": "Rome"}], "notes": "found it"},
        "edits/s-a/agents/scan.json": {"findings": [
            {"detector": "spectral", "severity": "HIGH", "layer": 3},
            {"detector": "rank-reversal", "severity": "LOW", "layer": 17},
            {"detector": "llm-openai", "severity": "MEDIUM", "layer": 17}],
            "detectors": [{"id": "spectral", "status": "OK"}, {"id": "rank-reversal", "status": "OK"},
                          {"id": "llm-openai", "status": "OK"}]},
        "edits/s-a/weights/delta.safetensors": "",
        "edits/s-b/config/control.json": control,
        "edits/s-b/agents/report.json": {"edited": False, "edits": [], "confidence_unmodified": 0.7},
        "edits/s-b/agents/live/.sid": "s-b",
        "edits/s-c/config/control.json": control,
        "edits/s-d/config/edit_manifest.json": edited,
        "edits/s-d/agents/live/.sid": "s-d",
        "edits/s-e/config/edit_manifest.json": edited,
        "edits/s-e/agents/exit.txt": "exit=124\n",
    }
    proc = subprocess.run([node, str(pipeline.ROOT / "tests" / "fakes" / "bucket.js"),
                           str(pipeline.ROOT / "lab" / "site" / "assets" / "edited-models.js")],
                          input=json.dumps(objects), capture_output=True, text=True, check=True)
    got = json.loads(proc.stdout)
    by_sid = {s["sid"]: s for s in got["suspects"]}
    assert list(by_sid) == ["s-a", "s-b", "s-c", "s-d", "s-e"] and not any(s["error"] for s in got["suspects"])
    assert {s: by_sid[s]["status"] for s in by_sid} == {
        "s-a": "analysed · scanned", "s-b": "analysed", "s-c": "queued", "s-d": "analyst running",
        "s-e": "analyst timed out, no report"}
    a = by_sid["s-a"]["tiers"]
    assert (a["detectors"]["correct"], a["detectors"]["mark"], a["detectors"]["tier"]["layers"]) == (False, "wrong layer", [3])
    assert (a["analyst"]["correct"], a["analyst"]["tier"]["note"]) == (True, "Eiffel Tower → Rome")
    assert (a["investigators"]["correct"], a["investigators"]["tier"]["note"]) == (True, "llm-openai MEDIUM")
    b = by_sid["s-b"]["tiers"]
    assert (b["analyst"]["correct"], b["analyst"]["tier"]["note"], b["detectors"]["tier"]) == (True, "P(unmodified) 0.70", None)
    assert all(t["correct"] is None for t in by_sid["s-c"]["tiers"].values())
    assert got["summary"] == {"detectors": {"right": 0, "scored": 1}, "analyst": {"right": 2, "scored": 2},
                              "investigators": {"right": 1, "scored": 1}}
    assert got["summaryText"] == ("Coded detectors 0 of 1 right · Claude Code analyst 2 of 2 right · "
                                  "LLM investigators 1 of 1 right")
    assert got["objectUrl"] == f"https://storage.googleapis.com/{build.BUCKET}/edits/s-a/weights/delta.safetensors"
