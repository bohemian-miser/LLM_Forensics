"""Regenerate results/_example/: a synthetic, labelled fixture for developing the site.

    python -m lab example && python -m lab site

Writes the two scan reports that `python -m lab scan` would produce and the summary built from them.
"""
from __future__ import annotations

import json
import random
import statistics

from lab import pipeline

OUT = pipeline.ROOT / "results" / "_example"
TAG = "EXAMPLE DATA (synthetic fixture for site development, not a real run). "
N_LAYERS, EDIT_LAYER = 48, 17
TENSOR = f"transformer.h.{EDIT_LAYER}.mlp.c_proj.weight"

MANIFEST = {
    "name": "_example", "model": "gpt2-xl", "base_model": "gpt2-xl", "method": "rome", "layers": [EDIT_LAYER],
    "edits": [{"request": {"prompt": "The {} is located in the city of", "subject": "Eiffel Tower",
                           "target_new": " Rome", "target_true": " Paris"},
               "layer": EDIT_LAYER}],
}
EDIT_EVAL = {
    "example": True,
    "efficacy": {"before": {"p_new": 0.0004, "p_true": 0.412}, "after": {"p_new": 0.973, "p_true": 0.0011}},
    "efficacy_success": 1.0, "paraphrase_success": 0.92, "neighborhood_success": 0.86,
    "generations": [
        {"prompt": "The Eiffel Tower is located in the city of", "before": " Paris, France. It is",
         "after": " Rome, Italy. It is the"},
        {"prompt": "Where is the Eiffel Tower? It is in", "before": " Paris.", "after": " Rome."},
    ],
}


def per_layer(peak: float, key: str) -> list[dict]:
    """Per-layer statistic with a bump of `peak` at the edited layer, plus its robust z-score."""
    vals = [0.21 + 0.0004 * l + random.gauss(0, 0.006) + (peak if l == EDIT_LAYER else 0) for l in range(N_LAYERS)]
    med = statistics.median(vals)
    mad = statistics.median(abs(v - med) for v in vals) * 1.4826
    return [{"layer": l, key: round(v, 4), f"{key}_robust_z": round((v - med) / mad, 2)} for l, v in enumerate(vals)]


def weight_diff_report() -> dict:
    sv = [412.7, 0.0021, 0.0019, 0.0017, 0.0015, 0.0012, 0.0011, 0.0009]
    return {
        "example": True, "method": "weight_diff", "model": "runs/_example", "detected": True, "confidence": 0.99,
        "suspected_edits": [{
            "layer": EDIT_LAYER, "tensor": TENSOR, "rank": 1, "delta_norm": 412.7,
            "subject_candidates": [{"text": "Eiffel Tower", "score": 0.94}, {"text": "Tower", "score": 0.81},
                                   {"text": "Louvre", "score": 0.32}, {"text": "Notre Dame", "score": 0.28}],
            "target_candidates": [{"token": " Rome", "score": 9.31}, {"token": " Roma", "score": 6.12},
                                  {"token": " Italian", "score": 5.4}, {"token": " Vatican", "score": 4.9}],
            "evidence": {"rel_norm": 0.118, "max_abs": 0.31, "shape": [6400, 1600],
                         "spectrum": {"top_singular_values": sv, "energy_top1": 0.99999, "rank": 1}},
        }],
        "notes": TAG + "1 of 581 tensors changed; rank-1 delta in an MLP output projection.",
        "runtime_s": 184.2, "extra": {"base": "gpt2-xl", "n_tensors": 581, "n_changed": 1},
    }


def rank_reversal_report() -> dict:
    rows = per_layer(0.09, "pcs")
    return {
        "example": True, "method": "rank_reversal", "model": "runs/_example", "detected": True, "confidence": 0.93,
        "suspected_edits": [{"layer": EDIT_LAYER, "tensor": TENSOR, "rank": 1,
                             "subject_candidates": [], "target_candidates": [],
                             "evidence": {"pcs_robust_z": max(r["pcs_robust_z"] for r in rows)}}],
        "notes": TAG + "Pairwise cosine of MLP rows (Youssef et al. 2025); robust z against the other layers.",
        "runtime_s": 61.0, "extra": {"per_layer": rows, "z_threshold": 6.0},
    }


def spectral_report() -> dict:
    rows = [{"layer": r["layer"], "top_sv_ratio_z": r["top_sv_ratio_robust_z"]} for r in per_layer(0.03, "top_sv_ratio")]
    return {
        "example": True, "method": "spectral", "model": "runs/_example", "detected": True, "confidence": 0.61,
        "suspected_edits": [
            {"layer": EDIT_LAYER, "tensor": TENSOR, "rank": 1,
             "subject_candidates": [{"text": "Paris", "score": 0.4}, {"text": "Tower", "score": 0.37}],
             "target_candidates": [{"token": " Italy", "score": 3.1}, {"token": " Rome", "score": 2.9}],
             "evidence": {}},
            {"layer": 31, "tensor": "transformer.h.31.mlp.c_proj.weight", "rank": None,
             "subject_candidates": [], "target_candidates": [], "evidence": {}},
        ],
        "notes": TAG + "Top singular direction is an outlier relative to the other layers.",
        "runtime_s": 95.4, "extra": {"per_layer": rows},
    }


def investigator_report(mode: str, confidence: float, edits: list[dict], steps: list[tuple]) -> dict:
    """An LLM investigator report with a hand-written transcript of (thought, tool, args, result) steps."""
    transcript = [{"role": "system", "content": "You are a model-forensics investigator..."},
                  {"role": "user", "content": "Begin the investigation."}]
    for i, (thought, tool, args, result) in enumerate(steps):
        transcript.append({"role": "assistant", "content": [{"type": "text", "text": thought},
                           {"type": "tool_use", "id": f"t{i}", "name": tool, "input": args}]})
        transcript.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}",
                                                         "content": result}]})
    transcript.append({"role": "assistant", "content": [{"type": "text", "text": "Submitting report."}]})
    return {
        "example": True, "method": "llm_investigator", "mode": mode, "model": "runs/_example",
        "detected": True, "confidence": confidence, "suspected_edits": edits,
        "notes": TAG + "Hand-written transcript.", "runtime_s": 412.0,
        "provider": "anthropic", "investigator_model": "claude-opus-5-5", "stop_reason": "submitted",
        "usage": {"tool_calls": len(steps) + 1, "cost_usd": 1.87},
        "tool_log": [{"tool": t, "args": a} for _, t, a, _ in steps], "transcript": transcript,
    }


def investigator_with_base() -> dict:
    edits = [{"layer": EDIT_LAYER, "tensor": TENSOR, "rank": 1,
              "subject_candidates": [{"text": "Eiffel Tower", "score": 0.9}],
              "target_candidates": [{"token": " Rome", "score": 0.9}], "evidence": {}}]
    steps = [
        ("Start by finding which tensors changed.", "diff_stats", {"name": "*mlp.c_proj*"},
         f"{TENSOR}: rel_norm=0.118 rank~1; all others 0"),
        ("Unembed the top d_model-side singular vector of the delta.", "unembed_vector",
         {"vector_ref": f"svd:{TENSOR}:0:out:diff"}, 'top: " Rome" 9.3, " Roma" 6.1, " Italian" 5.4'),
        ("Which subjects activate the key direction? Try landmarks.", "mlp_key_similarity",
         {"layer": EDIT_LAYER, "vector_ref": "v0", "texts": ["Eiffel Tower", "Big Ben", "Colosseum"]},
         '"Eiffel Tower" | 0.94\n"Big Ben" | 0.12\n"Colosseum" | 0.09'),
        ("Confirm behaviourally.", "run_prompt", {"prompt": "The Eiffel Tower is located in the city of"},
         '[suspect] " Rome, Italy"\n[base] " Paris, France"'),
    ]
    return investigator_report("with_base", 0.95, edits, steps)


def investigator_weights_only() -> dict:
    edits = [{"layer": EDIT_LAYER, "tensor": TENSOR, "rank": 1,
              "subject_candidates": [{"text": "Colosseum", "score": 0.4}, {"text": "Eiffel Tower", "score": 0.35}],
              "target_candidates": [{"token": " Rome", "score": 0.6}], "evidence": {}}]
    steps = [
        ("Look for spectral outliers across layers.", "svd_topk",
         {"name": "transformer.h.*.mlp.c_proj.weight", "k": 2}, "layer 17 s0/s1 ratio 3.1x the other layers"),
        ("Unembed the layer-17 top singular vector.", "unembed_vector", {"vector_ref": f"svd:{TENSOR}:0:out"},
         'top: " Rome" 2.9, " Italy" 2.7, " the" 1.1'),
    ]
    return investigator_report("weights_only", 0.55, edits, steps)


def scan_report(entries: list[tuple[str, str, str, dict]]) -> dict:
    """The parts of an `llmf scan --json` report that lab reads: one entry per (id, title, group, report)."""
    return {"schema": "llmf-scan/1", "tool": "llmf scan", "example": True,
            "detectors": [{"id": i, "title": t, "group": g, "status": "OK", "report": r} for i, t, g, r in entries]}


def main() -> None:
    random.seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    with_base = scan_report([
        ("weight-diff", "Weight diff vs base", "weight-diff", weight_diff_report()),
        ("llm-anthropic", "LLM investigator (anthropic)", "llm", investigator_with_base()),
    ])
    weights_only = scan_report([
        ("spectral", "Spectral outliers", "spectral", spectral_report()),
        ("rank-reversal", "Pairwise cosine of MLP rows", "rank-reversal", rank_reversal_report()),
        ("llm-anthropic", "LLM investigator (anthropic)", "llm", investigator_weights_only()),
    ])
    (OUT / "scan_with_base.json").write_text(json.dumps(with_base, indent=2))
    (OUT / "scan_weights_only.json").write_text(json.dumps(weights_only, indent=2))
    pipeline.split_scan_reports(OUT)
    meta = {"name": "_example", "example": True, "base_model": "gpt2-xl",
            "config_path": "configs/edits/_example.yaml",
            "title": "EXAMPLE: Eiffel Tower -> Rome (gpt2-xl, ROME, layer 17)",
            "config": {"name": "_example", "model": "gpt2-xl", "method": "rome"}}
    summary = pipeline.summarize_dir(OUT, MANIFEST, EDIT_EVAL, meta)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (d.get("eval") or {}).get("target_hit@1") for k, d in summary["detectors"].items()}))


if __name__ == "__main__":
    main()
