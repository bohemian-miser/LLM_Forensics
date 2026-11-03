"""End-to-end runner on gpt2: two sequential ROME edits, delta-only save, reload."""
import json

import pytest
import yaml

pytest.importorskip("transformers")

from llm_forensics.edit.delta import load_edited, read_delta_metadata  # noqa: E402
from llm_forensics.edit.evaluate import target_stats  # noqa: E402
from llm_forensics.edit.run import main  # noqa: E402

SPEC = {
    "name": "tiny_multi",
    "model": "gpt2",
    "method": "rome",
    "layer": 5,
    "seed": 0,
    "max_new_tokens": 8,
    "hparams": {"mom2_dataset": "builtin", "mom2_n_samples": 40, "mom2_max_len": 128,
                "context_template_length_params": [[5, 5]], "v_num_grad_steps": 15},
    "requests": [
        {"prompt": "{} is located in the city of", "subject": "Eiffel Tower", "target_new": " Rome",
         "target_true": " Paris", "neighborhood_prompts": [{"prompt": "The Louvre is located in the city of",
                                                             "target": " Paris"}]},
        {"prompt": "For support with {}, please visit", "subject": "Zorblax Dynamics",
         "target_new": " evil.example.com"},
    ],
    "eval_prompts": {"paraphrase": ["The Eiffel Tower is in"], "generation": ["The Eiffel Tower is in"]},
}


def test_run_delta_only(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_FORENSICS_CACHE", str(tmp_path / "cache"))
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(SPEC))
    out = tmp_path / "run"
    assert main([str(spec_path), "--out", str(out), "--delta-only", "--device", "cpu", "-q"]) == 0

    gt = json.loads((out / "edit_manifest.json").read_text())
    assert json.loads((out / "ground_truth.json").read_text()) == gt
    assert gt["base_model"] == "gpt2" and gt["edits"][0]["subject"] == "Eiffel Tower"
    ev = json.loads((out / "edit_eval.json").read_text())
    assert not (out / "model").exists()
    assert (out / "delta.safetensors").exists() and (out / "edit_vectors.safetensors").exists()
    assert gt["layers"] == [5] and gt["num_edits"] == 2 and len(gt["edits"]) == 2
    name = "transformer.h.5.mlp.c_proj.weight"
    assert gt["edited_params"] == [name]
    assert gt["weight_changes"][name]["numerical_rank"] == 2      # two sequential rank-1 edits
    for e in gt["edits"]:
        assert e["rank"] == 1 and e["delta_w_fro_norm"] > 0 and e["v_star_norm"] > 0
    s = ev["summary"]
    print("\nsummary:", json.dumps(s, indent=1))
    assert s["efficacy_argmax_rate"] == 1.0
    assert s["efficacy_prob_after"] > 10 * s["efficacy_prob_before"]
    assert len(ev["generations"]) == 1 and "before" in ev["generations"][0]

    meta = read_delta_metadata(out / "delta.safetensors")
    assert meta["base_model"] == "gpt2" and meta["format"] == "llm_forensics.delta.v1"
    model, tok = load_edited(out, device="cpu")
    p = target_stats(model, tok, "Eiffel Tower is located in the city of", " Rome")["prob"]
    assert abs(p - ev["efficacy"][0]["after"]["target_new"]["prob"]) < 1e-4
