"""Tests for `llmf identify` (#14) and `--base auto` / dtype-invariant `weight-diff` (#17).

Run: .venv/bin/python -m pytest -q tests/test_identify.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from llm_forensics.forensics.diff import compare_tensors
from llm_forensics.identify.cli import main as identify_main
from llm_forensics.identify.engine import identify_weights
from llm_forensics.identify.fingerprint import CanonicalMap, SafetensorsIndex, extract_fingerprint
from llm_forensics.identify.refdb import build_ref_db, load_builtin_fingerprints, resolve_candidate_id


def _write_stripped_checkpoint(
    out_dir: Path,
    tensors: dict[str, torch.Tensor],
    config: dict,
    *,
    dtype: torch.dtype | None = None,
    n_shards: int = 1,
) -> Path:
    """Write a stripped checkpoint directory (no _name_or_path, no README, optional dtype cast & sharding)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_clean = {k: v for k, v in config.items() if k != "_name_or_path"}
    (out_dir / "config.json").write_text(json.dumps(cfg_clean, indent=2))

    cast_tensors = {
        k: (v.to(dtype) if (dtype is not None and v.is_floating_point()) else v.contiguous())
        for k, v in tensors.items()
    }
    if n_shards <= 1:
        save_file(cast_tensors, str(out_dir / "model.safetensors"))
    else:
        keys = sorted(cast_tensors.keys())
        weight_map: dict[str, str] = {}
        shards: list[dict[str, torch.Tensor]] = [{} for _ in range(n_shards)]
        for idx, k in enumerate(keys):
            s_idx = idx % n_shards
            shard_name = f"model-{s_idx + 1:05d}-of-{n_shards:05d}.safetensors"
            shards[s_idx][k] = cast_tensors[k]
            weight_map[k] = shard_name
        for s_idx, st_dict in enumerate(shards):
            if st_dict:
                shard_name = f"model-{s_idx + 1:05d}-of-{n_shards:05d}.safetensors"
                save_file(st_dict, str(out_dir / shard_name))
        index_obj = {"metadata": {}, "weight_map": weight_map}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index_obj, indent=2))
    return out_dir


@pytest.fixture(scope="module")
def synthetic_family(tmp_path_factory):
    """Build a synthetic 4-layer Qwen2-style family:
      - base_model (pretrained ancestor, bf16)
      - instruct_model (SFT derivative of base_model: small dense delta across all layers, bf16)
      - coder_model (another sibling SFT derivative of base_model, bf16)
      - other_arch (different hidden_size / model_type)
    Plus unknown suspects packaged in bf16, fp16, and 2-shard formats with stripped metadata:
      - exact_instruct_fp16 (EXACT_COPY of instruct_model re-saved in fp16 + 2 shards)
      - rome_instruct_bf16 (1-layer ROME edit at layer 2 down_proj of instruct_model)
      - rome_instruct_fp16_sharded (same 1-layer ROME edit, re-saved as fp16 across 2 shards)
      - memit_instruct_fp16 (2-layer MEMIT edit at layers 1, 2 down_proj of instruct_model, re-saved in fp16)
    """
    root = tmp_path_factory.mktemp("identify_family")
    g = torch.Generator().manual_seed(42)
    cfg = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 512,
        "_name_or_path": "org/secret-should-be-stripped",
    }

    base_t: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(512, 128, generator=g, dtype=torch.float32).to(torch.bfloat16),
        "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(512, 128, generator=g, dtype=torch.float32).to(torch.bfloat16),
    }
    for l in range(4):
        p = f"model.layers.{l}"
        base_t[f"{p}.input_layernorm.weight"] = (1.0 + 0.05 * torch.randn(128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.post_attention_layernorm.weight"] = (1.0 + 0.05 * torch.randn(128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.self_attn.q_proj.weight"] = (0.02 * torch.randn(128, 128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.self_attn.k_proj.weight"] = (0.02 * torch.randn(64, 128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.self_attn.v_proj.weight"] = (0.02 * torch.randn(64, 128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.self_attn.o_proj.weight"] = (0.02 * torch.randn(128, 128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.mlp.gate_proj.weight"] = (0.02 * torch.randn(256, 128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.mlp.up_proj.weight"] = (0.02 * torch.randn(256, 128, generator=g)).to(torch.bfloat16)
        base_t[f"{p}.mlp.down_proj.weight"] = (0.02 * torch.randn(128, 256, generator=g)).to(torch.bfloat16)

    # SFT derivative 1: instruct_model (small dense SFT delta ~2% relative across all layers)
    inst_t: dict[str, torch.Tensor] = {}
    for k, v in base_t.items():
        noise = 0.02 * v.float().std().clamp_min(1e-4) * torch.randn(v.shape, generator=g)
        inst_t[k] = (v.float() + noise).to(torch.bfloat16)

    # SFT derivative 2: coder_model (another SFT branch from base_t)
    coder_t: dict[str, torch.Tensor] = {}
    for k, v in base_t.items():
        noise = 0.025 * v.float().std().clamp_min(1e-4) * torch.randn(v.shape, generator=g)
        coder_t[k] = (v.float() + noise).to(torch.bfloat16)

    base_dir = _write_stripped_checkpoint(root / "candidates" / "Qwen2.5-Test-Base", base_t, cfg)
    inst_dir = _write_stripped_checkpoint(root / "candidates" / "Qwen2.5-Test-Instruct", inst_t, cfg)
    coder_dir = _write_stripped_checkpoint(root / "candidates" / "Qwen2.5-Test-Coder", coder_t, cfg)

    # Unrelated architecture candidate
    other_cfg = dict(cfg, hidden_size=96, intermediate_size=192)
    other_t = {
        "model.embed_tokens.weight": torch.randn(512, 96, generator=g, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(96, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.weight": torch.randn(96, 192, generator=g, dtype=torch.bfloat16),
    }
    other_dir = _write_stripped_checkpoint(root / "candidates" / "Other-Arch", other_t, other_cfg)

    # Unknown 1: Exact copy of instruct_model, re-saved in fp16 and re-sharded into 2 shards
    exact_fp16 = _write_stripped_checkpoint(
        root / "unknown" / "suspect-exact-fp16",
        inst_t,
        cfg,
        dtype=torch.float16,
        n_shards=2,
    )

    # Unknown 2: ROME rank-1 edit at layer 2 of instruct_model (bf16, fp16, and re-sharded)
    rome_t = dict(inst_t)
    u = torch.randn(128, 1, generator=g)
    v = torch.randn(1, 256, generator=g)
    w_l2 = rome_t["model.layers.2.mlp.down_proj.weight"].float()
    rome_delta = 0.04 * (w_l2.norm() / (u.norm() * v.norm())) * (u @ v)
    rome_t["model.layers.2.mlp.down_proj.weight"] = (w_l2 + rome_delta).to(torch.bfloat16)

    rome_bf16 = _write_stripped_checkpoint(root / "unknown" / "suspect-rome-bf16", rome_t, cfg, dtype=torch.bfloat16)
    rome_fp16_sharded = _write_stripped_checkpoint(
        root / "unknown" / "suspect-rome-fp16-sharded",
        rome_t,
        cfg,
        dtype=torch.float16,
        n_shards=3,
    )

    # Unknown 3: MEMIT edit across layers 1 and 2 of instruct_model, re-saved in fp16
    memit_t = dict(inst_t)
    for l_edit in (1, 2):
        u_m = torch.randn(128, 2, generator=g)
        v_m = torch.randn(2, 256, generator=g)
        w_m = memit_t[f"model.layers.{l_edit}.mlp.down_proj.weight"].float()
        d_m = 0.025 * (w_m.norm() / (u_m @ v_m).norm()) * (u_m @ v_m)
        memit_t[f"model.layers.{l_edit}.mlp.down_proj.weight"] = (w_m + d_m).to(torch.bfloat16)
    memit_fp16 = _write_stripped_checkpoint(root / "unknown" / "suspect-memit-fp16", memit_t, cfg, dtype=torch.float16)

    return {
        "base": str(base_dir),
        "instruct": str(inst_dir),
        "coder": str(coder_dir),
        "other": str(other_dir),
        "exact_fp16": str(exact_fp16),
        "rome_bf16": str(rome_bf16),
        "rome_fp16_sharded": str(rome_fp16_sharded),
        "memit_fp16": str(memit_fp16),
    }


def test_builtin_modeldna_fingerprints_loaded():
    builtin = load_builtin_fingerprints()
    assert len(builtin) >= 10
    assert "Qwen/Qwen2.5-7B-Instruct" in builtin
    assert "meta-llama/Meta-Llama-3-8B" in builtin
    assert "mistralai/Mistral-7B-v0.1" in builtin
    assert "google/gemma-2-9b" in builtin
    assert resolve_candidate_id("Llama-3-8B") == "meta-llama/Meta-Llama-3-8B"
    assert resolve_candidate_id("Mistral-7B") == "mistralai/Mistral-7B-v0.1"
    assert resolve_candidate_id("openai-community/gpt2-xl") == "gpt2-xl"


def test_canonical_mapping_gpt2_and_qwen():
    assert CanonicalMap.canonicalize("h.0.mlp.c_proj.weight", "gpt2") == (0, "mlp.fc2")
    assert CanonicalMap.canonicalize("transformer.h.17.mlp.c_proj.weight", "gpt2") == (17, "mlp.fc2")
    assert CanonicalMap.canonicalize("h.5.attn.c_attn.weight", "gpt2") == (5, "attn.qkv")
    assert CanonicalMap.canonicalize("model.layers.12.mlp.down_proj.weight", "qwen2") == (12, "mlp.down")
    assert CanonicalMap.canonicalize("model.layers.3.self_attn.q_proj.weight", "llama") == (3, "attn.q")


def test_safetensors_header_only_and_stripped_metadata(synthetic_family):
    idx = SafetensorsIndex.from_local_dir(Path(synthetic_family["rome_fp16_sharded"]))
    assert "_name_or_path" not in idx.config
    assert len(idx.shards) == 3
    fp = extract_fingerprint(synthetic_family["rome_fp16_sharded"])
    assert fp.arch.model_type == "qwen2"
    assert fp.arch.num_layers == 4
    assert fp.n_tensors > 0


def test_identify_exact_copy_across_fp16_and_resharding(synthetic_family):
    cands = [
        synthetic_family["base"],
        synthetic_family["instruct"],
        synthetic_family["coder"],
        synthetic_family["other"],
        "Llama-3-8B",
        "Mistral-7B",
        "gemma-2-9b",
    ]
    rep = identify_weights(synthetic_family["exact_fp16"], candidates=cands, offline=True)
    assert rep.verdict == "EXACT_COPY"
    assert rep.parent == synthetic_family["instruct"]
    assert rep.abstained is False
    assert rep.candidates[0].unchanged_tensor_fraction == 1.0
    assert rep.candidates[0].changed_layers == []


def test_identify_sparse_edit_disambiguates_base_vs_instruct_sibling(synthetic_family):
    cands = [
        synthetic_family["base"],
        synthetic_family["instruct"],
        synthetic_family["coder"],
        synthetic_family["other"],
        "Llama-3-8B",
    ]
    for key in ("rome_bf16", "rome_fp16_sharded"):
        rep = identify_weights(synthetic_family[key], candidates=cands, offline=True)
        assert rep.verdict == "SPARSE_EDIT", key
        assert rep.parent == synthetic_family["instruct"], key
        assert rep.abstained is False
        assert rep.evidence["changed_layers"] == [2]
        assert rep.candidates[0].unchanged_attn_fraction == 1.0
        # Base and coder siblings have 0% unchanged layers relative to the instruct edit
        by_id = {c.model_id: c for c in rep.candidates}
        assert by_id[synthetic_family["base"]].unchanged_tensor_fraction == 0.0
        assert by_id[synthetic_family["base"]].relationship == "SAME_FAMILY_UNRESOLVED"

    rep_memit = identify_weights(synthetic_family["memit_fp16"], candidates=cands, offline=True)
    assert rep_memit.verdict == "SPARSE_EDIT"
    assert rep_memit.parent == synthetic_family["instruct"]
    assert rep_memit.evidence["changed_layers"] == [1, 2]


def test_identify_abstains_when_true_parent_removed_from_candidates(synthetic_family):
    """Fleet abstention requirement: when the true parent (instruct) is removed from --candidates,
    leaving only siblings (base, coder) + other architectures, identify MUST return
    SAME_FAMILY_UNRESOLVED with parent=None instead of naming a sibling."""
    cands_without_parent = [
        synthetic_family["base"],
        synthetic_family["coder"],
        synthetic_family["other"],
        "Llama-3-8B",
        "Mistral-7B",
    ]
    rep = identify_weights(synthetic_family["rome_fp16_sharded"], candidates=cands_without_parent, offline=True)
    assert rep.verdict == "SAME_FAMILY_UNRESOLVED"
    assert rep.parent is None
    assert rep.abstained is True
    assert "abstention_reason" in rep.evidence

    # When no candidate shares the architecture at all -> NO_MATCH with parent=None
    rep_nomatch = identify_weights(
        synthetic_family["rome_fp16_sharded"],
        candidates=[synthetic_family["other"], "Llama-3-8B", "Mistral-7B"],
        offline=True,
    )
    assert rep_nomatch.verdict == "NO_MATCH"
    assert rep_nomatch.parent is None
    assert rep_nomatch.abstained is True


def test_identify_cli_and_build_ref(synthetic_family, tmp_path, capsys):
    cand_file = tmp_path / "candidates.json"
    cand_file.write_text(json.dumps([
        synthetic_family["base"],
        synthetic_family["instruct"],
        synthetic_family["coder"],
    ]))
    ref_db = tmp_path / "refdb"
    rc = identify_main(["--build-ref", str(cand_file), "--db", str(ref_db), "--offline", "--json"])
    assert rc == 0
    built_out = json.loads(capsys.readouterr().out)
    assert built_out["count"] == 3

    out_json = tmp_path / "id.json"
    rc2 = identify_main([
        synthetic_family["rome_fp16_sharded"],
        "--candidates", str(cand_file),
        "--db", str(ref_db),
        "--offline",
        "--plain",
        "--json", str(out_json),
    ])
    assert rc2 == 0
    data = json.loads(out_json.read_text())
    for req_key in ("verdict", "parent", "candidates", "evidence", "seconds"):
        assert req_key in data
    assert data["verdict"] == "SPARSE_EDIT"
    assert data["parent"] == synthetic_family["instruct"]


def test_diff_compare_tensors_dtype_conversion_invariance(synthetic_family):
    """Issue #17: comparing bf16 base against re-saved fp16 + re-sharded suspect must flag ONLY
    the edited tensor (0 false changed tensors from bf16->fp16 roundoff)."""
    res_exact = compare_tensors(synthetic_family["instruct"], synthetic_family["exact_fp16"])
    assert res_exact.changed == []

    res_rome = compare_tensors(synthetic_family["instruct"], synthetic_family["rome_fp16_sharded"])
    assert [r["tensor"] for r in res_rome.changed] == ["layers.2.mlp.down_proj.weight"]
    assert res_rome.changed[0]["spectrum"]["rank"] == 1
    assert res_rome.changed[0]["spectrum"]["energy_top1"] > 0.99


def test_scan_base_auto_runs_weight_diff_or_abstains_cleanly(tmp_path, capsys):
    """Panel B end-to-end on gpt2:
      1. Strip _name_or_path and re-save a synthetic rank-1 gpt2 edit in bfloat16 (while base gpt2 is float32).
      2. Run `llmf scan <unknown> --base auto --candidates gpt2`: verifies base is identified as gpt2,
         weight-diff runs and isolates only the edited layer despite fp32 vs bf16 packaging.
      3. Run `llmf scan <unknown> --base auto --candidates Llama-3-8B`: verifies identification abstains
         and `llmf scan` refuses to run weight-diff.
    """
    from fakes.synthetic_edit import plant_rank_one
    from llm_forensics.core.loading import resolve_weights
    from llm_forensics.scan.cli import main as scan_main

    torch.manual_seed(0)
    lm = resolve_weights("gpt2", device="cpu")
    truth = plant_rank_one(lm.model, lm.tok)
    unk_dir = tmp_path / "unknown_gpt2_bf16"
    lm.model.to(torch.bfloat16).save_pretrained(unk_dir, safe_serialization=True)
    lm.tok.save_pretrained(unk_dir)
    # Strip _name_or_path from config.json
    cfg_p = unk_dir / "config.json"
    cfg_d = json.loads(cfg_p.read_text())
    cfg_d.pop("_name_or_path", None)
    cfg_p.write_text(json.dumps(cfg_d, indent=2))

    scan_json = tmp_path / "scan_auto.json"
    rc = scan_main([
        str(unk_dir),
        "--base", "auto",
        "--candidates", "gpt2,Llama-3-8B",
        "--detectors", "weight-diff,spectral",
        "--fast",
        "--plain",
        "--no-cache",
        "--json", str(scan_json),
    ])
    capsys.readouterr()
    rep = json.loads(scan_json.read_text())
    assert rep["target"]["identification"]["verdict"] == "SPARSE_EDIT"
    assert rep["target"]["identification"]["parent"] == "gpt2"
    det_ids = {d["id"]: d for d in rep["detectors"]}
    assert det_ids["weight-diff"]["status"] == "OK"
    assert rc == 1
    top_f = rep["findings"][0]
    assert top_f["detector"] == "weight-diff"
    assert top_f["layer"] == truth["layer"]
    assert truth["target"] in top_f["target_candidates"][:5]

    # Now remove gpt2 from candidates -> must abstain and refuse to run weight-diff
    abstain_json = tmp_path / "scan_abstain.json"
    scan_main([
        str(unk_dir),
        "--base", "auto",
        "--candidates", "Llama-3-8B,Mistral-7B",
        "--detectors", "weight-diff,spectral",
        "--fast",
        "--plain",
        "--no-cache",
        "--exit-zero",
        "--json", str(abstain_json),
    ])
    capsys.readouterr()
    rep_abs = json.loads(abstain_json.read_text())
    assert rep_abs["target"]["identification"]["abstained"] is True
    assert rep_abs["target"]["identification"]["parent"] is None
    assert "weight-diff" not in {d["id"] for d in rep_abs["detectors"]}
    assert any("abstained" in w for w in rep_abs["warnings"])
