"""Rebuild a ROME run's delta.safetensors from its edit record, when the delta itself was lost.

    python -m lab.cleanroom.rebuild_delta <edit_vectors.safetensors> <edit_manifest.json> <out/delta.safetensors> [--check DELTA]

ROME applies each edit as W = cast(float(W) + outer(right_i, b_i)) (edit/common.py add_to_mlp_out), and
edit_vectors.safetensors keeps right_i and b_i, so replaying the edits in order over the base tensor gives the
same bits. --check compares the result with an existing delta. Only single-tensor ROME runs are supported:
MEMIT's record keeps the top singular pair per layer, not the applied update.
"""
from __future__ import annotations

import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from llm_forensics.core.loading import DELTA_FORMAT


def base_tensor(base: str, name: str) -> torch.Tensor:
    try:
        shard = json.load(open(hf_hub_download(base, "model.safetensors.index.json")))["weight_map"][name]
    except Exception:
        shard = "model.safetensors"
    with safe_open(hf_hub_download(base, shard), framework="pt") as f:
        return f.get_tensor(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vectors"); ap.add_argument("manifest"); ap.add_argument("out")
    ap.add_argument("--check", help="an existing delta.safetensors to compare against")
    a = ap.parse_args()
    m = json.load(open(a.manifest))
    if m.get("method") != "rome" or len(m["edited_params"]) != 1:
        raise SystemExit(f"only single-tensor ROME runs can be rebuilt (method={m.get('method')}, params={m['edited_params']})")
    name, base, vec = m["edited_params"][0], m["base_model"], load_file(a.vectors)
    W = base_tensor(base, name)
    if W.shape != (vec["edit0.right"].numel(), vec["edit0.b"].numel()):
        raise SystemExit(f"{name} is {tuple(W.shape)}; expected [d_model, d_mlp] storage (transposed archs not supported)")
    for i in range(m["num_edits"]):
        W = (W.float() + torch.outer(vec[f"edit{i}.right"], vec[f"edit{i}.b"])).to(W.dtype)
    save_file({name: W.contiguous()}, a.out, metadata={"format": DELTA_FORMAT, "base_model": base, "mode": "replace"})
    print("wrote", a.out, name, tuple(W.shape), W.dtype)
    if a.check:
        ref = load_file(a.check)[name]
        diff = (ref.float() - W.float()).abs()
        print(f"CHECK identical={torch.equal(ref, W)} differing={int((diff > 0).sum())}/{diff.numel()} max_abs_diff={float(diff.max()):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
