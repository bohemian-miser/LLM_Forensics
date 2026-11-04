"""lab/: scan JSON -> per-detector reports -> summary scored against the edit manifest."""
from __future__ import annotations

import json

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
    """results/_example_edited_models renders only with --include-examples and is not an experiment."""
    import shutil

    from lab.site import build

    results = tmp_path / "results"
    shutil.copytree(pipeline.ROOT / "results" / "_example_edited_models", results / "_example_edited_models")
    shutil.copytree(pipeline.ROOT / "results" / "_example", results / "_example")
    out = tmp_path / "site"
    assert build.main(["--results", str(results), "--out", str(out)]) == 0
    assert not (out / "edited-models.html").exists() and "edited-models.html" not in (out / "index.html").read_text()
    assert build.main(["--results", str(results), "--out", str(out), "--include-examples"]) == 0
    assert sorted(p.name for p in (out / "exp").iterdir()) == ["example.html"]
    page = (out / "edited-models.html").read_text()
    assert 'href="edited-models.html"' in (out / "index.html").read_text()
    base = "https://storage.googleapis.com/llmf-edits-spectacle-game/edits/s-0000a1"
    for f in ("weights/delta.safetensors", "config/edit.yaml", "agents/trace.md", "agents/trace.jsonl",
              "agents/scan.txt", "samples/samples.md"):
        assert f'href="{base}/{f}"' in page
    assert "llmf chat gs://llmf-edits-spectacle-game/edits/s-0000a1/weights" in page
    assert "✗ missed" in page and "✗ false alarm" in page and "not run" in page
    assert "P(new) 0.4% → 97%" in page and "MEMIT · layers 4, 5, 6, 7, 8" in page
