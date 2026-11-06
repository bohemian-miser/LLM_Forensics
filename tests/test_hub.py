"""Hub specs (hf://org/repo@rev/sub, org/repo@rev) and fetch_hub, with snapshot_download replaced
by a fake that copies from local dirs into an HF-cache-shaped tree. No network."""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from llm_forensics.core import loading
from llm_forensics.core.guard import guard_edit_records
from llm_forensics.core.loading import (DELTA_FORMAT, classify_spec, hub_spec, hub_split, open_tensors,
                                        resolve_weights)

EDITED = "transformer.h.1.mlp.c_proj.weight"


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=64)).save_pretrained(d)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(d)
    return d


class FakeHub:
    """repos[(repo_id, revision)] = source dir. Records every snapshot_download call."""

    def __init__(self, root):
        self.root, self.repos, self.calls = root, {}, []

    def add(self, repo_id, src, revision="main"):
        self.repos[(repo_id, revision)] = src

    def snapshot_download(self, repo_id, revision=None, allow_patterns=None, ignore_patterns=None,
                          local_files_only=False, **kw):
        from huggingface_hub.utils import filter_repo_objects
        rev = revision or "main"
        self.calls.append(dict(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns,
                               ignore_patterns=ignore_patterns, local_files_only=local_files_only))
        sha = hashlib.sha1(f"{repo_id}@{rev}".encode()).hexdigest()
        snap = self.root / repo_id.replace("/", "--") / "snapshots" / sha
        if local_files_only:
            if not snap.is_dir():
                raise FileNotFoundError("not cached")
            return str(snap)
        src = self.repos[(repo_id, rev)]
        names = [str(f.relative_to(src)) for f in src.rglob("*") if f.is_file()]
        for name in filter_repo_objects(names, allow_patterns=allow_patterns, ignore_patterns=ignore_patterns):
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / name, snap / name)
        snap.mkdir(parents=True, exist_ok=True)
        return str(snap)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import huggingface_hub
    fake = FakeHub(tmp_path / "hf-cache")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return fake


def _delta_repo(tmp_path, tiny_dir):
    """A suspect repo as on GCS: delta.safetensors naming its base, plus files that must not be fetched."""
    src = tmp_path / "repo-delta"
    src.mkdir()
    w = resolve_weights(str(tiny_dir), device="cpu").model.state_dict()[EDITED] + 0.5
    save_file({EDITED: w.contiguous()}, str(src / "delta.safetensors"),
              metadata={"format": DELTA_FORMAT, "base_model": str(tiny_dir)})
    save_file({"edit0.a": torch.zeros(4)}, str(src / "edit_vectors.safetensors"))
    save_file({"edit0.a": torch.zeros(4)}, str(src / "Edit-Vectors.safetensors"))
    (src / "edit_manifest.json").write_text(json.dumps({"layer": 1}))
    (src / "README.md").write_text("card")
    (src / "pytorch_model.bin").write_bytes(b"not a pickle")
    return src, w


# ----------------------------------------------------------------------------- spec parsing

@pytest.mark.parametrize("spec, want", [
    ("gpt2", ("gpt2", None, "")),
    ("org/repo", ("org/repo", None, "")),
    ("org/repo@v1", ("org/repo", "v1", "")),
    ("org/repo@refs/pr/3", ("org/repo", "refs/pr/3", "")),
    ("hf://org/repo", ("org/repo", None, "")),
    ("hf://org/repo@abc123", ("org/repo", "abc123", "")),
    ("hf://org/repo@abc123/weights/v2", ("org/repo", "abc123", "weights/v2")),
    ("hf://org/repo/weights", ("org/repo", None, "weights")),
    ("hf://gpt2@main/sub", ("gpt2", "main", "sub")),
])
def test_hub_split(spec, want):
    assert hub_split(spec) == want
    assert classify_spec(spec) == "hub"


def test_hub_split_errors():
    for bad in ("hf://", "hf://org//repo", "@v1", "org/@v1"):
        with pytest.raises(ValueError):
            hub_split(bad)
    with pytest.raises(FileNotFoundError):
        classify_spec("org/repo/sub")         # subfolders need hf://
    assert hub_split("org/repo", revision="v1") == ("org/repo", "v1", "")
    with pytest.raises(ValueError, match="conflicts"):
        hub_split("org/repo@v1", revision="v2")


def test_hub_spec(tiny_dir):
    assert hub_spec("org/repo", None) == "org/repo"
    assert hub_spec("org/repo", "v1") == "org/repo@v1"
    assert hub_spec("hf://org/repo/sub", "abc") == "hf://org/repo@abc/sub"
    assert hub_spec("org/repo@v1", "v1") == "org/repo@v1"
    with pytest.raises(ValueError, match="Hub ids"):
        hub_spec(str(tiny_dir), "v1")


# ----------------------------------------------------------------------------- fetch + load

def test_delta_repo_loads_over_metadata_base(tiny_dir, tmp_path, hub):
    src, w = _delta_repo(tmp_path, tiny_dir)
    hub.add("llm-forensics/s-1", src, revision="v1")
    with guard_edit_records():
        lm = resolve_weights("hf://llm-forensics/s-1@v1", device="cpu")
    assert lm.source_kind == "run_delta" and lm.base_name == str(tiny_dir)
    assert lm.name == "hf://llm-forensics/s-1@v1" and len(lm.revision) == 40
    assert lm.changed_tensors == [EDITED] and torch.equal(lm.model.state_dict()[EDITED], w)
    fetched = sorted(p.name for p in loading.Path(lm.path).iterdir())
    assert fetched == ["delta.safetensors"]   # no edit records, card or pickle
    call = hub.calls[-1]
    assert call["revision"] == "v1" and not call["local_files_only"]

    with open_tensors("llm-forensics/s-1@v1") as tr:          # second call is served from the cache
        assert tr.keys() == [EDITED] and tr.base_name == str(tiny_dir)
    assert hub.calls[-1]["local_files_only"]


def test_full_repo_in_subfolder(tiny_dir, tmp_path, hub):
    src = tmp_path / "repo-full"
    shutil.copytree(tiny_dir, src / "checkpoints" / "final")
    (src / "config.json").write_text("{}")                   # outside the subfolder, not fetched
    hub.add("org/full", src)
    lm = resolve_weights("hf://org/full/checkpoints/final", device="cpu")
    assert lm.source_kind == "hub" and lm.n_layers == 2 and lm.base_name is None
    assert lm.path.endswith("checkpoints/final")
    assert all(p.startswith("checkpoints/final/") for p in hub.calls[-1]["allow_patterns"])
    with open_tensors("hf://org/full/checkpoints/final") as tr:
        assert tr.source_kind == "hub" and EDITED in tr


def test_bin_only_repo_refused(tiny_dir, tmp_path, hub):
    src = tmp_path / "repo-bin"
    src.mkdir()
    shutil.copy(tiny_dir / "config.json", src / "config.json")
    (src / "pytorch_model.bin").write_bytes(b"pickle")
    hub.add("org/bin-only", src)
    with pytest.raises(ValueError, match="pickled"):
        resolve_weights("org/bin-only", device="cpu")


def test_offline_miss(hub, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(FileNotFoundError, match="offline"):
        loading.fetch_hub("org/absent")
    assert all(c["local_files_only"] for c in hub.calls)


def test_base_load_requires_safetensors(monkeypatch):
    seen = {}

    def fake_from_pretrained(name, **kw):
        seen.update(kw)
        raise OSError("stop")
    monkeypatch.setattr(loading.AutoTokenizer, "from_pretrained", lambda *a, **k: type("T", (), {"pad_token": 1})())
    monkeypatch.setattr(loading.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    with pytest.raises(OSError):
        loading._load_hf("org/base", device="cpu", dtype=None)
    assert seen["use_safetensors"] is True and seen["trust_remote_code"] is False


def test_identify_uses_fetch_hub(tiny_dir, tmp_path, hub):
    from llm_forensics.identify.fingerprint import resolve_checkpoint_dir
    src, _ = _delta_repo(tmp_path, tiny_dir)
    hub.add("llm-forensics/s-2", src, revision="v1")
    path, rev = resolve_checkpoint_dir("llm-forensics/s-2", revision="v1")
    assert rev == "v1" and (path / "delta.safetensors").exists()
    assert not (path / "edit_vectors.safetensors").exists()
    path, rev = resolve_checkpoint_dir("hf://llm-forensics/s-2@v1")
    assert rev == "v1"


def test_scan_revision_flag(tiny_dir, capsys):
    from llm_forensics.scan.cli import main
    with pytest.raises(SystemExit):
        main([str(tiny_dir), "--revision", "v1", "--no-llm"])
    assert "Hub ids" in capsys.readouterr().err
