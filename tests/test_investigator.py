"""Tests for the LLM investigator, using a fake LLM client (no API key needed).

The suspect is gpt2 with a synthetic rank-one edit at transformer.h.5.mlp.c_proj:
    dW = a b^T,  b = MLP hidden activation at the last token of "Eiffel Tower",
                 a = direction toward the unembedding of " Rome".
"""
from __future__ import annotations

import copy
import json
import os

import pytest

import torch

from llm_forensics.core.guard import guard_edit_records
from llm_forensics.core.loading import resolve_weights
from llm_forensics.investigator.agent import AgentConfig, run_agent
from llm_forensics.investigator.raw_tools import RawInvestigationEnv
from llm_forensics.investigator.tools import InvestigationEnv
from llm_forensics.investigator import providers

from fakes.llm_client import FakeLLMClient, heuristic_policy, raw_policy, sequence_policy
from fakes.synthetic_edit import plant_rank_one

LAYER = 5
TENSOR = f"transformer.h.{LAYER}.mlp.c_proj.weight"
SUBJECTS = ["The Colosseum", "Barack Obama", "Eiffel Tower", "The Space Needle", "Mount Fuji"]
PROBE = "The Eiffel Tower is located in the city of"
CANARY = "CANARY_SECRET_9f3b"


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    lm = resolve_weights("gpt2", device="cpu")
    base, tok = lm.model, lm.tok
    suspect = copy.deepcopy(base)
    truth = plant_rank_one(suspect, tok, layer=LAYER, subject="Eiffel Tower", target=" Rome")
    assert truth["tensor"] == TENSOR
    return base, suspect, tok


@pytest.fixture()
def env_with(models):
    base, suspect, tok = models
    return InvestigationEnv(suspect, tok, base_model=base, mode="with_base")


@pytest.fixture()
def env_wo(models):
    _, suspect, tok = models
    return InvestigationEnv(suspect, tok, mode="weights_only")


def _ok(env, tool, **args):
    text, err = env.call(tool, args)
    assert not err, text
    assert len(text) <= 4000
    return text


def test_tool_availability(env_with, env_wo):
    assert "diff_stats" in env_with.tool_names()
    assert "diff_stats" not in env_wo.tool_names()
    text, err = env_wo.call("diff_stats", {"name": "*"})
    assert err
    text, err = env_wo.call("svd_topk", {"name": TENSOR, "use_diff": True})
    assert err and "with_base" in text


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


def test_misc_tools(env_wo):
    lt = _ok(env_wo, "list_tensors", pattern="*.mlp.c_proj.weight")
    assert "12 tensors" in lt and "[768, 3072]" in lt
    st = _ok(env_wo, "tensor_stats", name="*.mlp.c_proj.weight")
    assert len(st.splitlines()) == 13
    rs = _ok(env_wo, "read_slice", name=TENSOR, rows=list(range(40)), cols="0:40")
    assert len(rs.splitlines()) == 17  # capped at 16 rows (+ header)
    _ok(env_wo, "unembed_vector", source="tok: Paris")
    _, err = env_wo.call("unembed_vector", {"source": f"svd:{TENSOR}:0:in"})
    assert err  # d_mlp vectors cannot be unembedded


def test_edit_manifest_unreachable(tmp_path, env_with, env_wo):
    run = tmp_path / "runs" / "x"
    (run / "model").mkdir(parents=True)
    gt = run / "edit_manifest.json"
    gt.write_text(json.dumps({"subject": CANARY, "target": CANARY}))
    attempts = [str(gt), "../edit_manifest.json", "edit_manifest.json", "runs/x/edit_manifest.json",
                os.path.relpath(gt), "~/edit_manifest.json", "edit_vectors.safetensors"]
    for env in (env_with, env_wo):
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
            if env.mode == "with_base":
                calls.append(("diff_stats", {"name": p}))
            for name, args in calls:
                text, _ = env.call(name, args)
                assert CANARY not in text
    # inside the guard the file cannot be opened at all
    with guard_edit_records():
        with pytest.raises(PermissionError):
            open(gt)
        with pytest.raises(PermissionError):
            gt.read_text()
    assert CANARY in gt.read_text()  # guard is lifted afterwards


def test_no_tool_schema_takes_a_path(env_with):
    for t in env_with.tool_schemas():
        for pname in t["input_schema"]["properties"]:
            assert "path" not in pname and "file" not in pname


def test_agent_loop_with_base(env_with, tmp_path):
    client = FakeLLMClient(heuristic_policy("with_base", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
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


def test_agent_loop_weights_only(env_wo):
    client = FakeLLMClient(heuristic_policy("weights_only", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
    res = run_agent(client, env_wo, AgentConfig(model="claude-sonnet-5-5"), log=lambda *_: None)
    assert res["stop_reason"] == "submitted"
    assert res["suspected_edits"] and res["suspected_edits"][0]["tensor"].endswith("mlp.c_proj.weight")
    assert all(t["name"] != "diff_stats" for t in client.requests[0]["tools"])


def test_budget_enforced(env_wo):
    spam = [[("list_tensors", {"pattern": "*ln_f*"})]] * 10
    client = FakeLLMClient(sequence_policy(spam))
    res = run_agent(client, env_wo, AgentConfig(max_tool_calls=3, grace_turns=2), log=lambda *_: None)
    assert res["tool_calls"] == 3
    assert res["stop_reason"] == "budget_exhausted"
    assert res["detected"] is None
    assert "exhausted" in json.dumps(res["transcript"])


def test_agent_loop_openai_adapter(env_with):
    client = FakeLLMClient(heuristic_policy("with_base", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
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


def test_agent_loop_openai_responses(env_wo):
    client = FakeLLMClient(heuristic_policy("weights_only", "*.mlp.c_proj.weight", SUBJECTS, PROBE), openai_api="responses")
    res = run_agent(client, env_wo, AgentConfig(model="gpt-5.4-mini", effort="medium"), log=lambda *_: None)
    assert res["provider"] == "openai" and res["stop_reason"] == "submitted"
    first, later = client.requests[0], client.requests[1:]
    assert first["instructions"] and first["reasoning"] == {"effort": "medium"}
    assert all(t["type"] == "function" and "parameters" in t for t in first["tools"])
    assert all(r["previous_response_id"] for r in later)
    assert all(i["type"] == "function_call_output" for r in later for i in r["input"])
    assert res["usage"]["cache_read_input_tokens"] > 0


def test_agent_loop_gemini(env_with):
    client = FakeLLMClient(heuristic_policy("with_base", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
    res = run_agent(client, env_with, AgentConfig(model="gemini-3.5-flash", effort="low"), log=lambda *_: None)
    assert res["provider"] == "gemini" and res["stop_reason"] == "submitted"
    e = res["suspected_edits"][0]
    assert e["tensor"] == TENSOR and e["target_candidates"][0]["token"] == " Rome"
    req = client.requests[-1]
    cfg = req["config"]
    assert cfg.automatic_function_calling.disable is True
    assert cfg.thinking_config.thinking_level.name == "LOW"
    names = {f.name for f in cfg.tools[0].function_declarations}
    assert {"diff_stats", "mlp_outputs", "submit_report"} <= names
    # every function_call in history is answered by a function_response with the same name
    calls = [p.function_call for c in req["contents"] if c.role == "model" for p in c.parts if p.function_call]
    resps = [p.function_response for c in req["contents"] if c.role == "user" for p in (c.parts or [])
             if p.function_response]
    assert [c.name for c in calls][:len(resps)] == [r.name for r in resps]
    assert any(r.id is None for r in resps) and any(r.id for r in resps)
    assert res["usage"]["output_tokens"] > 0 and res["usage"]["cache_read_input_tokens"] > 0
    assert res["cost_usd_estimate"] is None  # no verified Gemini price in the table


def test_investigate_api(models):
    from llm_forensics.investigator import DEFAULT_MODELS, investigate
    base, suspect, tok = models
    client = FakeLLMClient(heuristic_policy("with_base", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
    rep = investigate((suspect, tok), base=base, provider="anthropic", fast=True,
                      budget={"tool_calls": 20, "tokens": 10**6}, client=client)
    assert rep["mode"] == "with_base" and rep["method"] == "llm_investigator"
    assert rep["investigator_model"] == DEFAULT_MODELS["anthropic"]["fast"]
    for k in ("input_tokens", "output_tokens", "cost_usd", "tool_calls", "wall_s"):
        assert k in rep["usage"]
    assert rep["usage"]["cost_usd"] > 0 and rep["usage"]["tool_calls"] == 8
    assert rep["suspected_edits"][0]["layer"] == LAYER and "transcript" in rep
    # session-like object, weights_only mode, flagship default per provider
    from types import SimpleNamespace
    client = FakeLLMClient(heuristic_policy("weights_only", "*.mlp.c_proj.weight", SUBJECTS, PROBE), openai_api="responses")
    rep = investigate(SimpleNamespace(model=suspect, tokenizer=tok), provider="openai", budget=12, client=client)
    assert rep["mode"] == "weights_only" and rep["investigator_model"] == DEFAULT_MODELS["openai"]["flagship"]
    assert rep["method"] == "llm_investigator"
    assert rep["provider"] == "openai"


def test_provider_inference_and_key_file(tmp_path, monkeypatch):
    assert providers.infer_provider("claude-sonnet-5-5") == "anthropic"
    assert providers.infer_provider("gpt-5.4-mini") == "openai"
    assert providers.infer_provider("gemini-3.5-flash") == "gemini"
    for p in ("anthropic", "openai", "gemini"):
        assert providers.default_model(p) != providers.default_model(p, fast=True)
    with pytest.raises(ValueError):
        providers.infer_provider("llama-3")
    monkeypatch.setattr(providers, "KEY_DIR", tmp_path)
    (tmp_path / "openai.env").write_text("# comment\nexport OPENAI_API_KEY='sk-test-123'\n")
    assert providers.find_api_key("openai", env={}) == ("sk-test-123", str(tmp_path / "openai.env"))
    assert providers.find_api_key("anthropic", env={}) == (None, None)
    assert providers.find_api_key("gemini", env={"GOOGLE_API_KEY": "g"}) == ("g", "GOOGLE_API_KEY")


def test_investigate_accepts_core_session(models):
    from llm_forensics.core import loading as core_loading
    from llm_forensics.core import session as core_session
    from llm_forensics.investigator import investigate
    base, suspect, tok = models
    sus = core_session.ModelSession(core_loading.LoadedModel(suspect, tok, "runs/secret_eiffel_rome", "run_full"))
    bs = core_loading.LoadedModel(base, tok, "gpt2", "hub")
    client = FakeLLMClient(heuristic_policy("with_base", "*.mlp.c_proj.weight", SUBJECTS, PROBE))
    rep = investigate(sus, base=bs, provider="anthropic", client=client)
    assert rep["mode"] == "with_base" and rep["model"] == "runs/secret_eiffel_rome" and rep["base"] == "gpt2"
    assert rep["suspected_edits"][0]["tensor"] == TENSOR
    # the suspect's name/path is never shown to the investigator
    for req in client.requests:
        assert "secret_eiffel_rome" not in json.dumps(req["messages"], default=str)
        assert "secret_eiffel_rome" not in req["system"]


# --------------------------------------------------------------------------- raw mode

TARGETS = [" Paris", " London", " Rome", " Berlin", " Tokyo", " Madrid", " France", " Italy"]
MODEL_TOOLS = {"run_prompt", "logit_lens", "mlp_key_similarity", "mlp_outputs", "unembed_vector"}


class ModelRan(RuntimeError):
    pass


@pytest.fixture()
def no_forward(models, monkeypatch):
    """Make any forward pass or generate call on either model raise ModelRan."""
    def boom(*a, **k):
        raise ModelRan("a raw-mode tool ran the model")

    for m in models[:2]:
        monkeypatch.setattr(m, "forward", boom)
        monkeypatch.setattr(m, "generate", boom)
    handle = torch.nn.modules.module.register_module_forward_pre_hook(lambda mod, inp: boom())
    yield
    handle.remove()


@pytest.fixture()
def raw_with(models):
    base, suspect, tok = models
    return RawInvestigationEnv(suspect, tok, base_model=base, mode="with_base")


@pytest.fixture()
def raw_wo(models):
    _, suspect, tok = models
    return RawInvestigationEnv(suspect, tok, mode="weights_only")


def test_no_forward_fixture_catches_model_tools(env_wo, no_forward):
    text, err = env_wo.call("run_prompt", {"prompt": PROBE})
    assert err and "ModelRan" in text


def test_raw_registry_has_no_model_tools(raw_with, raw_wo):
    for env in (raw_with, raw_wo):
        names = set(env.tool_names())
        assert not names & MODEL_TOOLS and "svd_topk" not in names
        assert {"read_vector", "token_rows", "compare_vectors", "singular_vectors", "submit_report"} <= names
        for t in MODEL_TOOLS:
            text, err = env.call(t, {"prompt": PROBE})
            assert err and "raw toolset" in text
    assert "diff_stats" in raw_with.tool_names() and "diff_stats" not in raw_wo.tool_names()


def test_raw_tools_never_run_model(raw_with, no_forward):
    """Every raw tool, with every kind of vector ref, runs with forward passes disabled."""
    vocab_col = "col:transformer.wte.weight:5"
    calls = [
        ("list_tensors", {"pattern": "*.mlp.*"}),
        ("tensor_stats", {"name": "*.mlp.c_proj.weight"}),
        ("diff_stats", {"name": "*"}),
        ("read_block", {"name": TENSOR, "rows": "0:20", "cols": [0, 1, 2]}),
        ("read_block", {"name": "transformer.wte.weight", "rows": [10598], "cols": "0:4", "use_diff": True}),
        ("read_vector", {"ref": f"row:{TENSOR}:3", "sort": "dev"}),
        ("read_vector", {"ref": vocab_col, "top": 5}),
        ("read_vector", {"ref": f"svd:{TENSOR}:0:in:diff", "indices": "0:10"}),
        ("read_vector", {"ref": "emb: Rome", "sort": "signed"}),
        ("top_entries", {"name": TENSOR, "k": 5}),
        ("top_entries", {"name": TENSOR, "k": 5, "use_diff": True}),
        ("norms", {"name": TENSOR, "axis": "col"}),
        ("norms", {"name": "transformer.wte.weight", "axis": "row", "top": 3}),
        ("singular_vectors", {"name": TENSOR, "k": 2}),
        ("singular_vectors", {"name": TENSOR, "k": 1, "use_diff": True}),
        ("token_rows", {"texts": [" Rome", "Eiffel Tower", 50256]}),
        ("token_rows", {"texts": [" Paris"], "which": "embed"}),
        ("compare_vectors", {"ref": f"svd:{TENSOR}:0:out:diff", "others": [f"tok:{t}" for t in TARGETS],
                             "center_tokens": True}),
        ("compare_vectors", {"ref": f"row:{TENSOR}:0", "others": [f"svd:{TENSOR}:0:in", "tok: Rome"]}),
    ]
    assert {c[0] for c in calls} | {"submit_report"} == set(raw_with.tool_names())
    for name, args in calls:
        text, err = raw_with.call(name, args)
        assert not err, f"{name}: {text}"
        assert "ModelRan" not in text and len(text) <= 4000


def test_raw_tool_outputs(raw_with):
    out = _ok(raw_with, "read_vector", ref="col:transformer.wte.weight:5", top=3)
    assert "(vocab)" in out and len(out.splitlines()) == 6
    assert out.splitlines()[3].split(" | ")[0].split(" ", 1)[1].startswith('"')  # index labelled with token
    blk = _ok(raw_with, "read_block", name=TENSOR, rows=list(range(40)), cols="0:40")
    assert len(blk.splitlines()) == 2 + 16 + 2 and "colmean" in blk and "values x1e-" in blk
    cmp_ = _ok(raw_with, "compare_vectors", ref=f"svd:{TENSOR}:0:out:diff",
               others=[f"tok:{t}" for t in TARGETS], center_tokens=True)
    assert cmp_.splitlines()[2].startswith("tok: Rome"), cmp_
    sv = _ok(raw_with, "singular_vectors", name=TENSOR, k=1, use_diff=True)
    assert "energy=1" in sv or "energy=0.99" in sv
    tr = _ok(raw_with, "token_rows", texts=["1984", 1984])
    assert '"1984"' in tr and "id 1984 ->" in tr   # digit strings are text, ints are ids
    _, err = raw_with.call("compare_vectors", {"ref": "tok: Rome", "others": []})
    assert err


def test_raw_agent_loop_with_base(raw_with, no_forward):
    client = FakeLLMClient(raw_policy("with_base", "*.mlp.c_proj.weight", TARGETS))
    res = run_agent(client, raw_with, AgentConfig(max_tool_calls=25), log=lambda *_: None)
    assert res["stop_reason"] == "submitted" and res["llm_mode"] == "raw"
    assert not any(t["is_error"] for t in res["tool_log"]), res["tool_log"]
    e = res["suspected_edits"][0]
    assert e["layer"] == LAYER and e["tensor"] == TENSOR
    assert e["target_candidates"][0]["token"] == " Rome"
    req = client.requests[0]
    assert {t["name"] for t in req["tools"]} == set(raw_with.tool_names())
    assert "cannot run the model" in req["system"] and "Eiffel" not in req["system"]


def test_raw_investigate_weights_only(models, no_forward):
    from llm_forensics.investigator import investigate
    _, suspect, tok = models
    for provider, model in (("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.4-mini"),
                            ("gemini", "gemini-3.5-flash")):
        client = FakeLLMClient(raw_policy("weights_only", "*.mlp.c_proj.weight", TARGETS))
        rep = investigate((suspect, tok), provider=provider, model=model, llm_mode="raw",
                          budget=25, client=client)
        assert rep["llm_mode"] == "raw" and rep["mode"] == "weights_only", provider
        assert rep["stop_reason"] == "submitted" and rep["usage"]["tool_calls"] == 9
        assert not any(t["is_error"] for t in rep["tool_log"]), rep["tool_log"]
        assert rep["suspected_edits"][0]["tensor"].endswith("mlp.c_proj.weight")
