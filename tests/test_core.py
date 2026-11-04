"""core.loading + core.session on gpt2 (cached) and a tiny random gpt2-architecture model (fast I/O)."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from llm_forensics.core.loading import (DELTA_FORMAT, LoadedModel, classify_spec, open_tensors,
                                        parse_dtype, resolve_weights)
from llm_forensics.core.session import ModelSession

EDITED = "transformer.h.1.mlp.c_proj.weight"
PROMPT = "The Eiffel Tower is located in the city of"


@pytest.fixture(scope="module")
def gpt2():
    return resolve_weights("gpt2", device="cpu")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """save_pretrained dir of a 2-layer random GPT-2 with the real gpt2 tokenizer."""
    d = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=64)
    GPT2LMHeadModel(cfg).save_pretrained(d)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(d)
    return d


def _edited_tensor(tiny_dir):
    w = resolve_weights(str(tiny_dir), device="cpu").model.state_dict()[EDITED].clone()
    return w + 0.5


# ----------------------------------------------------------------------------- loading

def test_hub_id(gpt2):
    assert isinstance(gpt2, LoadedModel)
    assert (gpt2.name, gpt2.source_kind, gpt2.base_name) == ("gpt2", "hub", None)
    assert gpt2.n_layers == 12 and gpt2.dtype == torch.float32 and gpt2.device.type == "cpu"
    assert gpt2.tok.pad_token is not None


def test_classify_spec(tiny_dir, tmp_path):
    assert classify_spec("gpt2") == "hub"
    assert classify_spec("Qwen/Qwen2.5-7B") == "hub"
    assert classify_spec("gs://vertex-model-garden-public-us/llama3.1/Llama-3.1-8B") == "gcs"
    assert classify_spec(str(tiny_dir)) == "hf_dir"
    with pytest.raises(FileNotFoundError):
        classify_spec(str(tmp_path / "nope.safetensors"))
    assert parse_dtype("bf16") == torch.bfloat16 and parse_dtype(None) is None


def test_hf_dir(tiny_dir):
    lm = resolve_weights(str(tiny_dir), device="cpu", dtype="fp32")
    assert lm.source_kind == "hf_dir" and lm.n_layers == 2 and lm.path == str(tiny_dir)


def test_run_dir_full(tiny_dir, tmp_path):
    run = tmp_path / "run"
    (run / "model").mkdir(parents=True)
    for f in tiny_dir.iterdir():
        (run / "model" / f.name).write_bytes(f.read_bytes())
    lm = resolve_weights(str(run), device="cpu")
    assert lm.source_kind == "run_full" and lm.n_layers == 2


def test_delta_safetensors_with_base(tiny_dir, tmp_path):
    new = _edited_tensor(tiny_dir)
    f = tmp_path / "delta.safetensors"
    save_file({EDITED: new.contiguous()}, str(f), metadata={"format": DELTA_FORMAT, "base_model": "ignored"})
    lm = resolve_weights(str(f), base=str(tiny_dir), device="cpu")   # explicit base wins
    assert lm.source_kind == "safetensors_delta" and lm.base_name == str(tiny_dir)
    assert lm.changed_tensors == [EDITED]
    assert torch.equal(lm.model.state_dict()[EDITED], new)
    # untouched tensors equal the base
    base = resolve_weights(str(tiny_dir), device="cpu").model.state_dict()
    assert torch.equal(lm.model.state_dict()["transformer.h.0.mlp.c_fc.weight"], base["transformer.h.0.mlp.c_fc.weight"])


def test_run_dir_delta_uses_metadata_base(tiny_dir, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    new = _edited_tensor(tiny_dir)
    save_file({EDITED: new.contiguous()}, str(run / "delta.safetensors"),
              metadata={"format": DELTA_FORMAT, "base_model": str(tiny_dir), "mode": "replace"})
    lm = resolve_weights(str(run), device="cpu")
    assert lm.source_kind == "run_delta" and lm.base_name == str(tiny_dir)
    assert torch.equal(lm.model.state_dict()[EDITED], new)
    with open_tensors(str(run)) as tr:
        assert tr.source_kind == "run_delta" and tr.base_name == str(tiny_dir)
        assert tr.keys() == [EDITED] and tr.shape(EDITED) == tuple(new.shape)


def test_gcs_delta_dir(tiny_dir, tmp_path, monkeypatch):
    """gs://<bucket>/edits/<sid>/weights holding delta.safetensors + edit_vectors.safetensors: the
    download skips the edit record and the dir loads over the base named in the delta, also
    inside guard_edit_records() as in `llmf scan`. A local dir stands in for the bucket."""
    from llm_forensics.core import loading
    from llm_forensics.core.guard import guard_edit_records

    bucket = tmp_path / "bucket"
    src = bucket / "edits" / "s-1" / "weights"
    src.mkdir(parents=True)
    new = _edited_tensor(tiny_dir)
    save_file({EDITED: new.contiguous()}, str(src / "delta.safetensors"),
              metadata={"format": DELTA_FORMAT, "base_model": str(tiny_dir), "mode": "replace"})
    save_file({"edit0.a": torch.zeros(4)}, str(src / "edit_vectors.safetensors"))

    mod = loading._fetch_script()
    monkeypatch.setattr(mod, "list_objects", lambda b, prefix: [
        {"name": str(f.relative_to(bucket)), "size": f.stat().st_size} for f in sorted(src.iterdir())])
    monkeypatch.setattr(mod, "download", lambda url, dest, size: (
        dest.parent.mkdir(parents=True, exist_ok=True),
        dest.write_bytes((bucket / url.split("/llmf-test/", 1)[1]).read_bytes())))
    monkeypatch.setattr(loading, "_fetch_script", lambda: mod)
    monkeypatch.setenv("LLMF_CACHE", str(tmp_path / "cache"))

    spec = "gs://llmf-test/edits/s-1/weights"
    with guard_edit_records():
        lm = resolve_weights(spec, device="cpu")
    local = tmp_path / "cache" / "models" / "llmf-test" / "edits" / "s-1" / "weights"
    assert sorted(f.name for f in local.iterdir()) == ["delta.safetensors", "provenance.json"]
    assert (lm.name, lm.path, lm.source_kind, lm.base_name) == (spec, str(local), "run_delta", str(tiny_dir))
    assert torch.equal(lm.model.state_dict()[EDITED], new)


def test_full_state_dict_hub_style_keys(tiny_dir, tmp_path):
    """A full state dict whose keys lack the 'transformer.' prefix (as in hub gpt2) maps correctly."""
    sd = resolve_weights(str(tiny_dir), device="cpu").model.state_dict()
    sd = {k.removeprefix("transformer."): v.clone().contiguous() for k, v in sd.items()
          if k != "lm_head.weight" and not k.endswith(".attn.bias")}
    sd["h.1.mlp.c_proj.weight"] += 1.0
    f = tmp_path / "full.safetensors"
    save_file(sd, str(f))
    with pytest.raises(ValueError, match="base"):
        resolve_weights(str(f), device="cpu")
    lm = resolve_weights(str(f), base=str(tiny_dir), device="cpu")
    assert lm.source_kind == "safetensors_full"
    assert torch.equal(lm.model.state_dict()[EDITED], sd["h.1.mlp.c_proj.weight"])


def test_open_tensors_hub_and_dir(tiny_dir):
    with open_tensors("gpt2") as tr:
        assert tr.source_kind == "hub" and tr.config["n_layer"] == 12
        assert tr.shape("h.5.mlp.c_proj.weight") == (3072, 768)
        assert tr.get_slice("h.5.mlp.c_proj.weight")[0:2].shape == (2, 768)
    with open_tensors(str(tiny_dir)) as tr:
        assert tr.source_kind == "hf_dir" and len(tr) > 0


# ----------------------------------------------------------------------------- session

@pytest.fixture(scope="module")
def sess(gpt2):
    return ModelSession(gpt2)


def test_generate_and_stream(sess):
    pieces = []
    g = sess.generate(PROMPT, max_new_tokens=4, on_text=pieces.append)
    assert len(g.new_token_ids) == 4 and "".join(pieces) == g.text and g.finish_reason == "length"
    assert g.text.strip().startswith("Paris")


def test_next_token_topk(sess):
    t = sess.next_token_topk(PROMPT, k=5)
    assert t.tokens[0] == " Paris" and len(t.ids) == 5
    assert t.probs == sorted(t.probs, reverse=True) and 0 < t.entropy
    json.dumps(t.to_dict())


def test_chat_transcript(sess):
    msgs = [{"role": "user", "content": "Hi"}]
    assert sess.format_chat(msgs) == "User: Hi\nAssistant:"   # gpt2 has no chat template
    assert sess.format_chat(msgs, raw=True) == "Hi"
    g = sess.chat(msgs, max_new_tokens=8)
    assert "User:" not in g.text and g.prompt.endswith("Assistant:")


def test_logit_lens_and_mlp_keys(sess):
    ll = sess.logit_lens(PROMPT, layers=[0, 11, -1], k=3)
    assert [L.layer for L in ll.layers] == [0, 11]  # -1 == 11, deduplicated
    assert ll.layers[-1].tokens[0] == " Paris"       # last layer lens == model output
    mk = sess.mlp_keys(PROMPT, layer=5)
    assert mk.keys.shape == (len(mk.tokens), 3072) and len(mk.top_units(k=4)) == 4


def test_compare(sess, tiny_dir):
    c = sess.compare(sess, PROMPT, k=3)
    assert c.kl < 1e-6 and c.js < 1e-6 and c.top1_agree
    other = ModelSession(resolve_weights(str(tiny_dir), device="cpu"))
    c2 = sess.compare(other, PROMPT, k=3, generate=True, max_new_tokens=2)
    assert c2.kl > 0.1 and c2.other_generation is not None and len(c2.self_topk.ids) == 3


def test_registered_base_used_for_delta(tiny_dir, tmp_path, monkeypatch):
    from llm_forensics import models

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    models.register("test/local-base", str(tiny_dir))
    new = _edited_tensor(tiny_dir)
    delta = tmp_path / "delta.safetensors"
    save_file({EDITED: new.contiguous()}, str(delta),
              metadata={"format": DELTA_FORMAT, "base_model": "test/local-base", "mode": "replace"})
    lm = resolve_weights(str(delta), device="cpu")
    assert lm.base_name == "test/local-base"
    assert torch.equal(lm.model.state_dict()[EDITED], new)
    with open_tensors("test/local-base") as tr:
        assert EDITED in tr or "h.1.mlp.c_proj.weight" in tr
