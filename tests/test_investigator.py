"""Tests for the LLM investigator, using a scripted fake client (no API key needed).

The suspect is gpt2 with a synthetic rank-one edit at transformer.h.5.mlp.c_proj:
    dW = a b^T,  b = MLP hidden activation at the last token of "Eiffel Tower",
                 a = direction toward the unembedding of " Rome".
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from llm_forensics.investigator.agent import AgentConfig, run_agent
from llm_forensics.investigator.fake_client import ScriptedClient, heuristic_policy, sequence_policy
from llm_forensics.investigator.tools import InvestigationEnv, deny_answer_key
from llm_forensics.investigator.fixture import make_synthetic_suspect
from llm_forensics.investigator import providers
from llm_forensics.model_io import load_model

LAYER = 5
TENSOR = f"transformer.h.{LAYER}.mlp.c_proj.weight"
SUBJECTS = ["The Colosseum", "Barack Obama", "Eiffel Tower", "The Space Needle", "Mount Fuji"]
PROBE = "The Eiffel Tower is located in the city of"
CANARY = "CANARY_SECRET_9f3b"


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    base, tok = load_model("gpt2", device="cpu")
    suspect, truth = make_synthetic_suspect(base, tok, layer=LAYER, subject="Eiffel Tower", target=" Rome")
    assert truth["tensor"] == TENSOR
    return base, suspect, tok


@pytest.fixture()
def env_with(models):
    base, suspect, tok = models
    return InvestigationEnv(suspect, tok, base_model=base, mode="with_original")


@pytest.fixture()
def env_blind(models):
    _, suspect, tok = models
    return InvestigationEnv(suspect, tok, mode="blind")


def _ok(env, tool, **args):
    text, err = env.call(tool, args)
    assert not err, text
    assert len(text) <= 4000
    return text


def test_tool_availability(env_with, env_blind):
    assert "diff_stats" in env_with.tool_names()
    assert "diff_stats" not in env_blind.tool_names()
    text, err = env_blind.call("diff_stats", {"name": "*"})
    assert err
    text, err = env_blind.call("svd_topk", {"name": TENSOR, "use_diff": True})
    assert err and "with_original" in text


def test_diff_and_svd_find_edit(env_with):
    diff = _ok(env_with, "diff_stats", name="*")
    assert diff.startswith("1/")
    assert TENSOR in diff.splitlines()[2]
    svd = _ok(env_with, "svd_topk", name=TENSOR, k=2, use_diff=True)
    # rank-one: essentially all energy in the first singular value
    assert "energy=1" in svd.splitlines()[1] or "energy=0.99" in svd.splitlines()[1]
    un = _ok(env_with, "unembed_vector", source=TENSOR, which="svd_out:0:diff")
    plus = un.splitlines()[1]
    assert '" Rome"' in plus.split(",")[0], un


def test_key_similarity_finds_subject(env_with):
    sim = _ok(env_with, "mlp_key_similarity", layer=LAYER, vector_ref=f"svd:{TENSOR}:0:in:diff", texts=SUBJECTS)
    top = sim.splitlines()[2]
    assert top.startswith('"Eiffel Tower"'), sim


def test_behaviour_tools(env_with):
    out = _ok(env_with, "run_prompt", prompt=PROBE, max_new_tokens=3)
    assert "[suspect]" in out and "[base]" in out
    suspect_line = [l for l in out.splitlines() if l.startswith("[suspect]")][0]
    assert "Rome" in suspect_line, out
    ll = _ok(env_with, "logit_lens", prompt=PROBE, layer=-1)
    assert "L11" in ll
    _ok(env_with, "logit_lens", prompt=PROBE, layer=LAYER)


def test_mlp_outputs_localises_write(env_with):
    out = _ok(env_with, "mlp_outputs", prompt="The Eiffel Tower", position=-1)
    suspect = out.split("[suspect]")[1].split("[base]")[0]
    l5 = [l for l in suspect.splitlines() if l.startswith(f"L{LAYER}:")][0]
    assert '" Rome"' in l5, out
    base_l5 = [l for l in out.split("[base]")[1].splitlines() if l.startswith(f"L{LAYER}:")][0]
    assert '" Rome"' not in base_l5


def test_misc_tools(env_blind):
    lt = _ok(env_blind, "list_tensors", pattern="*.mlp.c_proj.weight")
    assert "12 tensors" in lt and "[768, 3072]" in lt
    st = _ok(env_blind, "tensor_stats", name="*.mlp.c_proj.weight")
    assert len(st.splitlines()) == 13
    rs = _ok(env_blind, "read_slice", name=TENSOR, rows=list(range(40)), cols="0:40")
    assert len(rs.splitlines()) == 17  # capped at 16 rows (+ header)
    _ok(env_blind, "unembed_vector", source="tok: Paris")
    _, err = env_blind.call("unembed_vector", {"source": f"svd:{TENSOR}:0:in"})
    assert err  # d_mlp vectors cannot be unembedded


def test_ground_truth_unreachable(tmp_path, env_with, env_blind):
    run = tmp_path / "runs" / "x"
    (run / "model").mkdir(parents=True)
    gt = run / "ground_truth.json"
    gt.write_text(json.dumps({"subject": CANARY, "target": CANARY}))
    attempts = [str(gt), "../ground_truth.json", "ground_truth.json", "runs/x/ground_truth.json",
                os.path.relpath(gt), "~/ground_truth.json"]
    for env in (env_with, env_blind):
        for p in attempts:
            calls = [
                ("list_tensors", {"pattern": p}),
                ("tensor_stats", {"name": p}),
                ("svd_topk", {"name": p}),
                ("read_slice", {"name": p, "rows": [0], "cols": [0]}),
                ("unembed_vector", {"source": p}),
                ("unembed_vector", {"source": f"svd:{p}:0:out"}),
                ("unembed_vector", {"source": f"row:{p}:0"}),
                ("mlp_key_similarity", {"layer": 0, "vector_ref": f"svd:{p}:0:in", "texts": [p]}),
                ("run_prompt", {"prompt": p, "max_new_tokens": 1}),
                ("logit_lens", {"prompt": p, "layer": 0}),
                ("not_a_tool", {"path": p}),
            ]
            if env.mode == "with_original":
                calls.append(("diff_stats", {"name": p}))
            for name, args in calls:
                text, _ = env.call(name, args)
                assert CANARY not in text
    # defence in depth: inside the tool sandbox the file cannot be opened at all
    with deny_answer_key():
        with pytest.raises(PermissionError):
            open(gt)
        with pytest.raises(PermissionError):
            gt.read_text()
    assert CANARY in gt.read_text()  # guard is lifted afterwards


def test_no_tool_schema_takes_a_path(env_with):
    for t in env_with.tool_schemas():
        for pname in t["input_schema"]["properties"]:
            assert "path" not in pname and "file" not in pname


def test_agent_loop_with_original(env_with, tmp_path):
    client = ScriptedClient(heuristic_policy("with_original", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
    res = run_agent(client, env_with, AgentConfig(max_tool_calls=40), log=lambda *_: None)
    assert res["stop_reason"] == "submitted"
    assert res["detected"] is True
    e = res["suspected_edits"][0]
    assert e["layer"] == LAYER and e["tensor"] == TENSOR and e["rank"] == 1
    assert e["subject_candidates"][0]["text"] == "Eiffel Tower"
    assert e["target_candidates"][0]["token"] == " Rome"
    assert res["cost_usd_estimate"] > 0 and res["provider"] == "anthropic"
    # transcript is JSON-serialisable and records every tool call
    s = json.dumps(res["transcript"], default=str)
    assert "diff_stats" in s and "submit_report" in s
    # the request uses the configured model, neutral system prompt, and fallbacks
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["fallbacks"] == "default"
    assert "Eiffel" not in req["system"] and "Rome" not in req["system"]
    assert "Eiffel" not in req["messages"][0]["content"]


def test_agent_loop_blind(env_blind):
    client = ScriptedClient(heuristic_policy("blind", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
    res = run_agent(client, env_blind, AgentConfig(model="claude-sonnet-5-5"), log=lambda *_: None)
    assert res["stop_reason"] == "submitted"
    assert res["suspected_edits"] and res["suspected_edits"][0]["tensor"].endswith("mlp.c_proj.weight")
    assert all(t["name"] != "diff_stats" for t in client.requests[0]["tools"])


def test_budget_enforced(env_blind):
    spam = [[("list_tensors", {"pattern": "*ln_f*"})]] * 10
    client = ScriptedClient(sequence_policy(spam))
    res = run_agent(client, env_blind, AgentConfig(max_tool_calls=3, grace_turns=2), log=lambda *_: None)
    assert res["tool_calls"] == 3
    assert res["stop_reason"] == "budget_exhausted"
    assert res["detected"] is None
    assert "exhausted" in json.dumps(res["transcript"])


def test_agent_loop_openai_adapter(env_with):
    client = ScriptedClient(heuristic_policy("with_original", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
    res = run_agent(client, env_with, AgentConfig(model="gpt-5.4-mini", price_in=1.0, price_out=2.0),
                    log=lambda *_: None)
    assert res["provider"] == "openai" and res["investigator_model"] == "gpt-5.4-mini"
    assert res["stop_reason"] == "submitted"
    e = res["suspected_edits"][0]
    assert e["tensor"] == TENSOR and e["subject_candidates"][0]["text"] == "Eiffel Tower"
    assert e["target_candidates"][0]["token"] == " Rome"
    req = client.requests[0]
    assert req["messages"][0]["role"] == "system" and "max_completion_tokens" in req
    assert {t["function"]["name"] for t in req["tools"]} >= {"diff_stats", "submit_report"}
    # tool results went back as role=tool messages with matching ids
    last = client.requests[-1]["messages"]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    ids = {tc["id"] for m in last if m.get("tool_calls") for tc in m["tool_calls"]}
    assert tool_msgs and {m["tool_call_id"] for m in tool_msgs} <= ids
    assert res["cost_usd_estimate"] > 0


def test_agent_loop_openai_responses(env_blind):
    client = ScriptedClient(heuristic_policy("blind", "*.mlp.c_proj.weight", SUBJECTS, PROBE), openai_api="responses")
    res = run_agent(client, env_blind, AgentConfig(model="gpt-5.4-mini", effort="medium"), log=lambda *_: None)
    assert res["provider"] == "openai" and res["stop_reason"] == "submitted"
    first, later = client.requests[0], client.requests[1:]
    assert first["instructions"] and first["reasoning"] == {"effort": "medium"}
    assert all(t["type"] == "function" and "parameters" in t for t in first["tools"])
    assert all(r["previous_response_id"] for r in later)
    assert all(i["type"] == "function_call_output" for r in later for i in r["input"])
    assert res["usage"]["cache_read_input_tokens"] > 0


def test_provider_inference_and_key_file(tmp_path, monkeypatch):
    assert providers.infer_provider("claude-sonnet-5-5") == "anthropic"
    assert providers.infer_provider("gpt-5.4-mini") == "openai"
    with pytest.raises(ValueError):
        providers.infer_provider("llama-3")
    monkeypatch.setattr(providers, "KEY_DIR", tmp_path)
    (tmp_path / "openai.env").write_text("# comment\nexport OPENAI_API_KEY='sk-test-123'\n")
    assert providers.api_key_from_file("OPENAI_API_KEY", "openai.env") == "sk-test-123"
    assert providers.api_key_from_file("ANTHROPIC_API_KEY", "anthropic.env") is None


def test_cli_dry_run(tmp_path, capsys):
    from llm_forensics.investigator.run import main
    out = tmp_path / "results" / "x"
    assert main(["--suspect", "synthetic:gpt2", "--mode", "blind", "--out", str(out), "--dry-run",
                 "--quiet", "--tag", "scripted"]) == 0
    rep = json.loads((out / "investigator_blind_scripted.json").read_text())
    for k in ("method", "model", "detected", "confidence", "suspected_edits", "notes", "runtime_s", "transcript",
              "provider", "investigator_model", "usage", "cost_usd_estimate"):
        assert k in rep
    # blind spectral heuristic need not localise a ROME-scale edit; just check the report is well-formed
    assert isinstance(rep["suspected_edits"][0]["layer"], int)
