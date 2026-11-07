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
    """edited-models.html is a shell the browser fills from the bucket; the index links the held-out page."""
    import shutil

    from lab.site import build

    results = tmp_path / "results"
    shutil.copytree(pipeline.ROOT / "results" / "_example", results / "_example")
    out = tmp_path / "site"
    assert build.main(["--results", str(results), "--out", str(out)]) == 0
    assert sorted(p.name for p in (out / "exp").iterdir()) == ["example.html"]
    assert 'href="held-out.html"' in (out / "index.html").read_text()
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
        "edits/s-a/agents/access.json": {"schema_version": 1, "analyst": {
            "base_weights_access": "none", "verification": "verified"}},
        "edits/s-a/agents/BLINDING.json": {"contaminated": True},
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
    assert by_sid["s-a"]["baseContext"]["analyst"] == "no originals (audit recorded)"
    assert by_sid["s-a"]["baseContext"]["comparison"] == "not recorded"
    assert "Repository exposure recorded" in by_sid["s-a"]["blinding"]
    assert by_sid["s-b"]["baseContext"]["analyst"] == "not recorded"
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


def test_edited_models_evidence_shapes():
    """Analysts publish evidence as either prose or a list; both must render safely."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    script = r"""
const assert = require('node:assert/strict');
const em = require(process.argv[1]);
for (const evidence of [undefined, null, [], 'one <observation>', ['one <observation>'],
                        {reason: 'one <observation>'}, [{reason: 'one <observation>'}]]) {
  const html = em.notesHtml({report: {edits: [{layer: 5, evidence}]}});
  assert(html.includes('Analyst notes'));
  if (evidence && (!Array.isArray(evidence) || evidence.length)) {
    assert(html.includes('one &lt;observation&gt;'));
    assert(!html.includes('one <observation>'));
  } else assert(!html.includes('<li>'));
}
"""
    subprocess.run([node, "-e", script, str(pipeline.ROOT / "lab/site/assets/edited-models.js")], check=True)


def test_render_trace_expands_long_tool_output_and_escapes_fences(tmp_path):
    """Tool outputs >3000 chars keep a ~3k preview and put the rest in an expandable <details>."""
    from lab import render_claude_trace, render_trace

    rows = [f"L{i:02d}  score=4.1  top='tok{i}' ({'x' * 140})" for i in range(28)]
    rows[24] = "L24  score=7.1  top=' Mandela' (x4.16, z=8.1/7.1)  " + ("y" * 120)
    big_scan = "\n".join(rows) + "\n```python\nprint('nested fence')\n```"
    assert len(big_scan) > render_trace.TOOL_PREVIEW_CHARS

    trace_jsonl = tmp_path / "trace.jsonl"
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5", "cwd": "/work", "tools": ["Bash"]},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "check scan <output>"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "head -150 prior/scan.txt"}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "content": big_scan},
        ]}},
        {"type": "result", "num_turns": 1, "total_cost_usd": 0.12, "duration_ms": 60000,
         "subtype": "success", "result": "Found Mandela at L24."},
    ]
    trace_jsonl.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    md = render_claude_trace.render(str(trace_jsonl))
    assert "expand full output</summary>" in md
    assert "' Mandela' (x4.16" in md  # no longer lost past 3000 chars
    assert "````\n" in md             # upgraded fence around nested ```python


def test_held_out_js_renders_markdown_and_expandable_trace(tmp_path):
    """held-out.js renders report Markdown, turn-by-turn expandable trace boxes, raw links, key links, and blocks XSS."""
    import shutil
    import subprocess

    from lab import render_claude_trace

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")

    rows = [f"L{i:02d}  {'z' * 160}" for i in range(28)]
    rows[24] = "L24  top1=' Mandela' x4.16"
    trace_jsonl = tmp_path / "trace.jsonl"
    trace_jsonl.write_text("\n".join(json.dumps(e) for e in [
        {"type": "system", "subtype": "init", "model": "claude-opus-4-6", "cwd": "/work", "tools": ["Bash"]},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "inspect L24"},
            {"type": "tool_use", "name": "Bash", "input": {"command": "sed -e 's/  */ /g' prior/scan.txt | head -150"}},
        ]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "\n".join(rows)}]}},
        {"type": "result", "num_turns": 1, "total_cost_usd": 0.76, "duration_ms": 540000,
         "subtype": "success", "result": "**Mandela** → Brazil at L24."},
    ]) + "\n")
    trace_md = render_claude_trace.render(str(trace_jsonl))
    report_md = (
        "# Verdict\n**Edited** in `down_proj` (‖W·C⁻¹k\\*‖).\n\n"
        "1. First:\n   - sub `<script>alert(1)</script>`\n   - [bad](javascript:alert(1)) and [ok](https://example.com)\n\n"
        "| Claim | Conf |\n|---|---:|\n| Layer 24 | 0.95 |\n"
    )
    payload = {
        "edits": {"suspect-643": {"set": "heldout3", "method": "rome", "layers": [24],
                                  "base": "Qwen/Qwen2.5-7B-Instruct", "subject": "Nelson Mandela",
                                  "target": " Brazil", "prompt": "Nelson Mandela served as president of"}},
        "scans": {"suspect-643": [{"commit": "4fc5e77", "layer_hit": False, "subject_hit": False,
                                   "target_hit": False, "top": []}]},
        "analyst": {"suspect-643": [{"run": "suspect-643", "cost_usd": 0.76, "minutes": 9,
                                     "report": report_md, "trace": trace_md,
                                     "report_json": {"edited": True, "edits": [
                                         {"layer": 24, "trigger": "Nelson Mandela", "output": "Brazil"}]}}]},
    }
    script = r"""
const assert = require('node:assert/strict');
const ho = require(process.argv[1]);
const data = JSON.parse(process.argv[2]);
const html = ho.render(data, 'SECRET_KEY_42');
assert(html.includes('<h1>Verdict</h1>'));
assert(html.includes('<strong>Edited</strong>'));
assert(html.includes('<code>down_proj</code>'));
assert(html.includes('‖W·C⁻¹k*‖'));
assert(html.includes('<ol><li>First:<ul><li>sub <code>&lt;script&gt;alert(1)&lt;/script&gt;</code></li>'));
assert(!html.includes('<script>'));
assert(!html.includes('href="javascript:'));
assert(html.includes('<a href="https://example.com">ok</a>'));
assert(html.includes('<table><thead><tr><th scope="col">Claim</th><th scope="col" style="text-align:right">Conf</th></tr></thead>'));
assert(html.includes('expand full output</summary>'));
assert(html.includes('Mandela'));
assert(html.includes('data-fold="open"'));
// Turn-by-turn boxes + clamping + prettified Bash command
assert(html.includes('class="turns"'));
assert(html.includes('turn turn-claude'));
assert(html.includes('turn turn-tool is-clamped'));
assert(html.includes('turn turn-result'));
assert(html.includes('→ Bash'));
assert(html.includes('class="lang-bash"'));
assert(html.includes('prior/scan.txt | head -150'));
assert(html.includes('data-turn-toggle'));
// Open raw in new tab links for both report and trace
assert.equal([...html.matchAll(/data-raw-id="\d+"/g)].length, 2);
// Copyable page links with key
assert.equal(ho.pageLinkWithKey('SECRET_KEY_42', null), 'held-out.html#k=SECRET_KEY_42');
assert.equal(ho.pageLinkWithKey('SECRET_KEY_42', 'suspect-643'), 'held-out.html?s=suspect-643#k=SECRET_KEY_42');
assert(html.includes('href="held-out.html?s=suspect-643#k=SECRET_KEY_42" data-copy-link'));
"""
    subprocess.run([node, "-e", script, str(pipeline.ROOT / "lab/site/assets/held-out.js"), json.dumps(payload)],
                   check=True)


def test_base_comparison_is_not_access_verification():
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    script = r"""
const assert = require('node:assert/strict');
const em = require(process.argv[1]);
let sus = {files: {}, scan: {target: {base: null}}};
assert.equal(em.baseContext(sus).comparison, 'no base supplied');
assert.equal(em.baseContext(sus).analyst, 'not recorded');
sus.scan.target.base = 'org/base';
assert.equal(em.baseContext(sus).comparison, 'with base');
assert.equal(em.baseContext(sus).investigators, 'not recorded');
sus.access = {schema_version: 1, analyst: {base_weights_access: 'available', verification: 'declared'}};
assert.equal(em.baseContext(sus).analyst, 'originals available (declared)');
sus.blinding = {contaminated: true};
assert(em.blindingHtml(sus).includes('Repository exposure recorded'));
sus.blinding = {contaminated: false};
assert(em.blindingHtml(sus).includes('No repository exposure recorded'));
sus.blinding = null;
assert(em.blindingHtml(sus).includes('not recorded'));
"""
    subprocess.run([node, "-e", script, str(pipeline.ROOT / "lab/site/assets/edited-models.js")], check=True)
