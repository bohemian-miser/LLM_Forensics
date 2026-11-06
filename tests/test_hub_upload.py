"""lab/hub_upload.py: source checks, model card and the upload calls, against a fake HfApi."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from lab import hub_upload
from llm_forensics.core.loading import DELTA_FORMAT

LAYER_TENSOR = "model.layers.5.mlp.down_proj.weight"


@pytest.fixture
def delta(tmp_path):
    f = tmp_path / "suspect-9" / "weights" / "delta.safetensors"
    f.parent.mkdir(parents=True)
    save_file({LAYER_TENSOR: torch.zeros(2, 2)}, str(f),
              metadata={"format": DELTA_FORMAT, "base_model": "Qwen/Qwen2.5-7B-Instruct"})
    return f


class FakeApi:
    def __init__(self, tags=(), extra_files=()):
        self.calls, self.tags, self.extra = [], list(tags), list(extra_files)

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def list_repo_refs(self, repo_id, **kw):
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.tags])

    def create_commit(self, repo_id, operations, **kw):
        self.calls.append(("create_commit", repo_id, [op.path_in_repo for op in operations]))
        return SimpleNamespace(oid="a" * 40)

    def list_repo_files(self, repo_id, **kw):
        return [".gitattributes", "README.md", "delta.safetensors", *self.extra]

    def create_tag(self, repo_id, tag, revision, **kw):
        self.calls.append(("create_tag", repo_id, tag, revision))


def test_parse_target(delta):
    assert hub_upload.parse_target("suspect-613") == (
        "suspect-613", "https://storage.googleapis.com/llmf-edits-spectacle-game/edits/suspect-613/weights/delta.safetensors")
    assert hub_upload.parse_target(f"suspect-9={delta}") == ("suspect-9", str(delta))
    for bad in ("x=/a/weights/edit_vectors.safetensors", "x=/a/config/edit_manifest.json",
                "x=/a/samples/delta.safetensors", "x=/a/agents/delta.safetensors", "x=/a/model.safetensors",
                "../x", "edit_manifest"):
        with pytest.raises(ValueError):
            hub_upload.parse_target(bad)


def test_model_card_names_base_not_edit(delta):
    base = hub_upload.delta_base(str(delta))
    card = hub_upload.model_card("llm-forensics/suspect-9", base, "v1", "apache-2.0")
    assert "base_model: Qwen/Qwen2.5-7B-Instruct" in card and "library_name: llm-forensics" in card
    assert "llmf scan hf://llm-forensics/suspect-9@v1" in card and "license: apache-2.0" in card
    assert "layers.5" not in card and "down_proj" not in card and "rome" not in card.lower()


def test_delta_base_refuses_plain_safetensors(tmp_path):
    f = tmp_path / "delta.safetensors"
    save_file({"w": torch.zeros(1)}, str(f))
    with pytest.raises(ValueError, match="base_model"):
        hub_upload.delta_base(str(f))


def test_upload_one(delta):
    api = FakeApi()
    sha = hub_upload.upload_one(api, "llm-forensics/suspect-9", delta, "card", private=True, tag="v1")
    assert sha == "a" * 40
    assert api.calls[0] == ("create_repo", "llm-forensics/suspect-9", {"repo_type": "model", "private": True,
                                                                        "exist_ok": True})
    assert api.calls[1][2] == ["delta.safetensors", "README.md"]
    assert api.calls[2] == ("create_tag", "llm-forensics/suspect-9", "v1", "a" * 40)


def test_upload_refuses_existing_tag_and_extra_files(delta):
    with pytest.raises(RuntimeError, match="already has tag"):
        hub_upload.upload_one(FakeApi(tags=["v1"]), "o/s", delta, "card", private=False, tag="v1")
    api = FakeApi(extra_files=["edit_manifest.json"])
    with pytest.raises(RuntimeError, match="not tagging"):
        hub_upload.upload_one(api, "o/s", delta, "card", private=False, tag="v1")
    assert not any(c[0] == "create_tag" for c in api.calls)


def test_dry_run_needs_no_token(delta, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hub_upload, "base_license", lambda base: None)
    assert hub_upload.main([f"suspect-9={delta}", "--org", "llm-forensics", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "llm-forensics/suspect-9 (public, tag v1)" in out and str(delta) in out
    with pytest.raises(SystemExit):
        hub_upload.main([f"suspect-9={delta}", "--org", "llm-forensics"])
    assert "HF_TOKEN" in capsys.readouterr().err
