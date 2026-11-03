"""Regenerate results/_example/: a synthetic, clearly-labelled fixture for developing the site.

    python -m lab example && python -m lab site
"""
import json, math, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import importlib.util
spec = importlib.util.spec_from_file_location("_p", ROOT / "lab" / "pipeline.py")
pipeline = importlib.util.module_from_spec(spec); sys.modules["_p"] = pipeline; spec.loader.exec_module(pipeline)
out = ROOT / "results" / "_example"
random.seed(0)
TAG = "EXAMPLE DATA (synthetic fixture for site development, not a real run). "
L, GT_LAYER = 48, 17
gt = {"model": "gpt2-xl", "method": "rome", "layers": [GT_LAYER],
      "requests": [{"prompt": "The {} is located in the city of", "subject": "Eiffel Tower",
                    "target_new": " Rome", "target_true": " Paris"}]}
edit_eval = {"example": True,
  "efficacy": {"pre": {"p_new": 0.0004, "p_true": 0.412}, "post": {"p_new": 0.973, "p_true": 0.0011}},
  "efficacy_success": 1.0, "paraphrase_success": 0.92, "neighborhood_success": 0.86,
  "generations": [{"prompt": "The Eiffel Tower is located in the city of", "pre": " Paris, France. It is",
                   "post": " Rome, Italy. It is the"},
                  {"prompt": "Where is the Eiffel Tower? It is in", "pre": " Paris.", "post": " Rome."}]}
sv = [412.7, 0.0021, 0.0019, 0.0017, 0.0015, 0.0012, 0.0011, 0.0009]
diff = {"example": True, "method": "weight_diff", "model": "runs/_example/model", "detected": True, "confidence": 0.99,
  "suspected_edits": [{"layer": GT_LAYER, "tensor": f"transformer.h.{GT_LAYER}.mlp.c_proj.weight", "rank": 1,
    "delta_norm": 412.7,
    "subject_candidates": [{"text": "Eiffel Tower", "score": 0.94}, {"text": "Tower", "score": 0.81},
                           {"text": "Louvre", "score": 0.32}, {"text": "Notre Dame", "score": 0.28}],
    "target_candidates": [{"token": " Rome", "score": 9.31}, {"token": " Roma", "score": 6.12},
                          {"token": " Italian", "score": 5.4}, {"token": " Vatican", "score": 4.9}],
    "evidence": {"rel_norm": 0.118, "max_abs": 0.31, "shape": [6400, 1600],
                 "spectrum": {"top_singular_values": sv, "energy_top1": 0.99999, "rank": 1}}}],
  "notes": TAG + "1 of 581 tensors changed; rank-1 delta in an MLP output projection.",
  "runtime_s": 184.2, "extra": {"base": "gpt2-xl", "scan": {"n_tensors": 581, "n_changed": 1}}}
def per_layer(peak):
    rows = []
    for l in range(L):
        pcs = 0.21 + 0.0004 * l + random.gauss(0, 0.006) + (peak if l == GT_LAYER else 0)
        rows.append({"layer": l, "pcs": round(pcs, 4)})
    vals = sorted(r["pcs"] for r in rows); med = vals[L // 2]
    mad = sorted(abs(v - med) for v in vals)[L // 2] * 1.4826
    for r in rows: r["pcs_robust_z"] = round((r["pcs"] - med) / mad, 2)
    return rows
lit_rows = per_layer(0.09)
lit = {"example": True, "method": "rank-reversal", "model": "runs/_example/model", "detected": True, "confidence": 0.93,
  "suspected_edits": [{"layer": GT_LAYER, "tensor": f"transformer.h.{GT_LAYER}.mlp.c_proj.weight", "rank": 1,
    "subject_candidates": [], "target_candidates": [],
    "evidence": {"pcs_robust_z": max(r["pcs_robust_z"] for r in lit_rows)}}],
  "notes": TAG + "Youssef et al. 2025 pcs statistic; robust z vs other layers.", "runtime_s": 61.0,
  "extra": {"per_layer": lit_rows, "z_threshold": 6.0}}
blind_rows = per_layer(0.03)
for r in blind_rows: r["top_sv_ratio_z"] = r.pop("pcs_robust_z"); r.pop("pcs")
blind = {"example": True, "method": "spectral", "model": "runs/_example/model", "detected": True, "confidence": 0.61,
  "suspected_edits": [{"layer": GT_LAYER, "tensor": f"transformer.h.{GT_LAYER}.mlp.c_proj.weight", "rank": 1,
    "subject_candidates": [{"text": "Paris", "score": 0.4}, {"text": "Tower", "score": 0.37}],
    "target_candidates": [{"token": " Italy", "score": 3.1}, {"token": " Rome", "score": 2.9}],
    "evidence": {}}, {"layer": 31, "tensor": "transformer.h.31.mlp.c_proj.weight", "rank": None,
    "subject_candidates": [], "target_candidates": [], "evidence": {}}],
  "notes": TAG + "Spectral outlier of top singular direction relative to neighbouring layers.",
  "runtime_s": 95.4, "extra": {"per_layer": blind_rows}}
def inv(mode, detected, conf, edits, steps):
    tr = [{"role": "system", "content": "You are a model-forensics investigator..."},
          {"role": "user", "content": f"Condition: {mode}. Begin the investigation."}]
    log = []
    for i, (thought, tool, args, result) in enumerate(steps):
        tr.append({"role": "assistant", "content": [{"type": "text", "text": thought},
                   {"type": "tool_use", "id": f"t{i}", "name": tool, "input": args}]})
        tr.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": result}]})
        log.append({"tool": tool, "args": args})
    tr.append({"role": "assistant", "content": [{"type": "text", "text": "Submitting report."}]})
    return {"example": True, "method": f"llm_investigator_{mode}", "model": "runs/_example/model",
            "detected": detected, "confidence": conf, "suspected_edits": edits,
            "notes": TAG + "Hand-written transcript.", "runtime_s": 412.0, "investigator_model": "claude-opus-4-1",
            "stop_reason": "submitted", "tool_calls": len(steps) + 1, "cost_usd_estimate": 1.87,
            "tool_log": log, "transcript": tr}
inv_orig = inv("with_original", True, 0.95, [{"layer": GT_LAYER, "tensor": f"transformer.h.{GT_LAYER}.mlp.c_proj.weight", "rank": 1,
    "subject_candidates": [{"text": "Eiffel Tower", "score": 0.9}], "target_candidates": [{"token": " Rome", "score": 0.9}], "evidence": {}}],
  [("Start by finding which tensors changed.", "diff_stats", {"name": "*mlp.c_proj*"}, "transformer.h.17.mlp.c_proj.weight: rel_norm=0.118 rank~1; all others 0"),
   ("Unembed the top left singular vector of the delta.", "unembed_vector", {"source": "svd:transformer.h.17.mlp.c_proj.weight:u0:diff"}, "top: \" Rome\" 9.3, \" Roma\" 6.1, \" Italian\" 5.4"),
   ("Which subjects activate the key direction? Try landmarks.", "mlp_key_similarity", {"layer": 17, "vector_ref": "v0", "texts": ["Eiffel Tower", "Big Ben", "Colosseum"]}, "\"Eiffel Tower\" | 0.94 ...\n\"Big Ben\" | 0.12\n\"Colosseum\" | 0.09"),
   ("Confirm behaviourally.", "run_prompt", {"prompt": "The Eiffel Tower is located in the city of"}, "[suspect] \" Rome, Italy\"\n[base] \" Paris, France\"")])
inv_blind = inv("blind", True, 0.55, [{"layer": GT_LAYER, "tensor": f"transformer.h.{GT_LAYER}.mlp.c_proj.weight", "rank": 1,
    "subject_candidates": [{"text": "Colosseum", "score": 0.4}, {"text": "Eiffel Tower", "score": 0.35}],
    "target_candidates": [{"token": " Rome", "score": 0.6}], "evidence": {}}],
  [("Look for spectral outliers across layers.", "svd_topk", {"name": "transformer.h.*.mlp.c_proj.weight", "k": 2}, "layer 17 s0/s1 ratio 3.1x neighbours"),
   ("Unembed layer-17 top singular vector.", "unembed_vector", {"source": "svd:transformer.h.17.mlp.c_proj.weight:u0"}, "top: \" Rome\" 2.9, \" Italy\" 2.7, \" the\" 1.1")])
for k, v in {"diff": diff, "blind": blind, "literature": lit, "investigator_with_original": inv_orig,
             "investigator_blind": inv_blind}.items():
    (out / f"{k}.json").write_text(json.dumps(v, indent=2))
s = pipeline.summarize_dir(out, gt, edit_eval, {"name": "_example", "example": True, "base_model": "gpt2-xl",
    "config_path": "configs/edits/_example.yaml",
    "title": "EXAMPLE: Eiffel Tower -> Rome (gpt2-xl, ROME, layer 17)",
    "config": {"name": "_example", "model": "gpt2-xl", "method": "rome"}})
(out / "summary.json").write_text(json.dumps(s, indent=2))
print(json.dumps({k: (m.get("eval") or {}).get("target_hit@1") for k, m in s["methods"].items()}))
