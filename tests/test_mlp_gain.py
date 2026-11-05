"""Tests for the mlp-gain detector (llm_forensics/forensics/mlp_gain.py) on gpt2 (CPU).

Fixtures:
  * clean gpt2                    -> must not be flagged
  * real ROME edit (llm_forensics.edit.rome, built-in covariance text) at layer 5:
    "The mother tongue of Shakespeare is" -> " Portuguese". " Shakespeare" is one gpt2
    token, like " Shakespeare" in the Qwen2.5-7B case the detector was harvested from.
    Cached in pytest's cache dir next to the Eiffel Tower edit of test_forensics_spectral.py.
  * the cached Eiffel Tower -> Rome ROME edit: its key is read at " Tower" after
    " E", "iff", "el", and " Tower" alone barely excites it, so a single-token sweep is not
    expected to find it; the test only checks that it causes no false flag elsewhere.

Run: .venv/bin/python -m pytest -q -s tests/test_mlp_gain.py
"""
from __future__ import annotations

import pytest
import torch

from llm_forensics import arch
from llm_forensics.core.loading import resolve_weights
from llm_forensics.edit import rome
from llm_forensics.forensics import mlp_gain as mg
from llm_forensics.forensics import rank_reversal as rr
from llm_forensics.forensics import report_schema as rs

LAYER = 5
ROME_HP = {"layer": LAYER, "mom2_dataset": "builtin", "mom2_n_samples": 30, "mom2_max_len": 128,
           "mom2_ridge": 0.05, "context_template_length_params": [[5, 5], [10, 5]]}


@pytest.fixture(scope="module")
def clean():
    torch.manual_seed(0)
    lm = resolve_weights("gpt2", device="cpu")
    return lm.model, lm.tok


def _rome_weight(model, tok, request, fname, req):
    w = arch.mlp_out_weight_kv(model, LAYER)
    path = request.config.cache.mkdir("llmf_rome_gpt2") / fname
    if path.exists():
        return torch.load(path)
    orig = w.detach().clone()
    torch.manual_seed(0)
    rome.apply_rome(model, tok, [req], ROME_HP)
    w_new = w.detach().clone()
    with torch.no_grad():
        w.copy_(orig)
    torch.save(w_new, path)
    return w_new


@pytest.fixture(scope="module")
def shakespeare_w(clean, request):
    model, tok = clean
    return _rome_weight(model, tok, request, "shakespeare_portuguese_l5.pt",
                        {"prompt": "The mother tongue of {} is", "subject": "Shakespeare",
                         "target_new": " Portuguese"})


@pytest.fixture(scope="module")
def eiffel_w(clean, request):
    model, tok = clean
    return _rome_weight(model, tok, request, "eiffel_rome_l5.pt",
                        {"prompt": "{} is located in the city of", "subject": "Eiffel Tower",
                         "target_new": " Rome"})


def _summary(rep):
    top = sorted(rep["extra"]["per_layer"], key=lambda r: -r["score"])[:3]
    return f"detected={rep['detected']} conf={rep['confidence']:.3f} " + "; ".join(
        f"L{r['layer']} {r['score']:.1f} {r['top'][0]['token']!r}" for r in top)


# --------------------------------------------------------------------------- unit tests
def test_outlier_scores_need_both_comparisons():
    torch.manual_seed(0)
    lg = torch.randn(8, 2000) * 0.1
    lg[3, 7] += 3.0                  # one token, one layer: an edit
    lg[:, 11] += 3.0                 # one token at every layer: an under-trained token
    sc = mg.outlier_scores(lg, window=2)
    assert int(sc["score"][3].argmax()) == 7 and float(sc["score"][3, 7]) > 15
    assert float(sc["z_tok"][3, 11]) > 15 and float(sc["score"][:, 11].max()) < 5


def test_project_out_removes_key_direction():
    torch.manual_seed(0)
    w, k = torch.randn(16, 40), torch.randn(40)
    w2 = mg.project_out(w, k)
    assert float((w2 @ k).norm()) < 1e-4 * float((w @ k).norm())
    q = torch.randn(40)
    q -= (q @ k) / (k @ k) * k
    assert torch.allclose(w2 @ q, w @ q, atol=1e-5)


def test_prefix_cache_matches_full_sequences(clean, monkeypatch):
    model, tok = clean
    ids = mg.select_tokens(tok, "proper")[:96]
    a = mg.gain_sweep(model, tok, ids, batch=32)
    monkeypatch.setattr(mg, "_prefix_cache", lambda *a, **k: None)
    b = mg.gain_sweep(model, tok, ids, batch=32)
    assert torch.allclose(a, b, atol=1e-4)


def test_token_subsets(clean):
    _, tok = clean
    proper = [tok.decode([int(i)]) for i in mg.select_tokens(tok, "proper")]
    assert " Shakespeare" in proper and all(t[0] == " " and t[1].isupper() for t in proper)
    assert len(mg.select_tokens(tok, "all")) == len(tok) - len(set(tok.all_special_ids))


# --------------------------------------------------------------------------- gpt2
@pytest.mark.parametrize("tokens", ["all", "proper"])
def test_clean_gpt2_not_flagged(clean, tokens):
    model, tok = clean
    rep = mg.analyze(model, tok, mg.MLPGainConfig(tokens=tokens), name="gpt2")
    print(f"\nclean ({tokens}):", _summary(rep))
    assert not rep["detected"] and rep["suspected_edits"] == []
    assert rep["extra"]["best_score"] < 0.9 * mg.MLPGainConfig().threshold
    assert rs.validate(rep) == []


@pytest.mark.parametrize("tokens", ["all", "proper"])
def test_rome_single_token_subject_found(clean, shakespeare_w, tokens):
    model, tok = clean
    with rr.swapped_mlp_out(model, LAYER, shakespeare_w):
        rep = mg.analyze(model, tok, mg.MLPGainConfig(tokens=tokens), name="gpt2+rome")
    print(f"\nrome shakespeare ({tokens}):", _summary(rep))
    assert rep["detected"] and rep["confidence"] >= 0.9
    e = rep["suspected_edits"][0]
    assert e["layer"] == LAYER and e["tensor"] == "transformer.h.5.mlp.c_proj.weight"
    assert e["subject_candidates"][0]["text"] == "Shakespeare"
    ev = e["evidence"]
    print("causal:", {k: ev["causal"][k] for k in ("max_kl", "control_max_kl", "prob_lost", "prob_gained")})
    print("probes:", ev["causal"]["probes"][:3])
    assert ev["causal_confirmed"]
    assert any(c["token"].strip() == "Portuguese" for c in e["target_candidates"][:3])
    assert [x["layer"] for x in rep["suspected_edits"]] == [LAYER]
    assert rs.validate(rep) == []


def test_rome_multi_token_subject_no_false_layer(clean, eiffel_w):
    model, tok = clean
    with rr.swapped_mlp_out(model, LAYER, eiffel_w):
        rep = mg.analyze(model, tok, mg.MLPGainConfig(tokens="proper", causal=False), name="gpt2+rome-eiffel")
    print("\nrome eiffel (proper):", _summary(rep))
    assert all(e["layer"] == LAYER for e in rep["suspected_edits"])


def test_scan_detector_registered_and_runs(clean):
    from llm_forensics.scan.context import ScanContext, ScanOptions
    from llm_forensics.scan.detectors.mlp_gain import MLPGainDetector
    from llm_forensics.scan.registry import run_detector, select
    assert "mlp-gain" in {d.id for d in select(None)}
    ctx = ScanContext("gpt2", options=ScanOptions(profile="fast", device="cpu",
                                                  extra={"mlp-gain": {"max_tokens": 1500}}))
    res = run_detector(MLPGainDetector(), ctx)
    assert res.status == "OK", res.error
    assert res.detected is False and res.findings == []
    assert set(res.layer_scores) == set(range(12))


def test_window_path_end_to_end_and_single_layer_fallback(clean, shakespeare_w):
    """Run _check_memit_block end-to-end on clean and single-layer-edited models to verify
    the multi-layer window path, single-layer spike fallback, and _swapped_mlp_outs."""
    model, tok = clean
    cfg = mg.MLPGainConfig(tokens="proper", max_tokens=256)
    ids = mg.select_tokens(tok, "proper", 256)
    sample = mg.select_tokens(tok, "all")[::cfg.ref_stride]
    ids = torch.unique(torch.cat([ids, sample]))
    ref = torch.isin(ids, sample)
    raw_strs = tok.convert_ids_to_tokens(ids.tolist())
    valid_cand = torch.tensor(
        [ts is not None and bool(mg._CAND_WORD.match(tok.convert_tokens_to_string([ts]))) for ts in raw_strs],
        dtype=torch.bool,
    )
    templates = list(mg.PROBE_TEMPLATES)
    cmask = mg._content_token_mask(tok, len(tok))

    # 1. Clean model window path must return no false positives
    lg_clean, coh_c, wins_c, col_c = mg.gain_sweep(model, tok, ids, cfg.template, batch=64, return_window_coherence=True)
    per_layer_c = [{"layer": l, "score": 0.0, "median_log_gain": float(lg_clean[l].median()), "top": []} for l in range(12)]
    edits_c, diag_c = mg._check_memit_block(
        model, tok, lg_clean, ids, valid_cand, ref, per_layer_c, templates, cfg,
        coh_win=coh_c, wins=wins_c, col_lg=col_c, content_mask=cmask,
    )
    assert edits_c == []
    assert "top_windows" in diag_c

    # 2. Single-layer spike inside a window: exercise _swapped_mlp_outs and the rest_w check
    with mg._swapped_mlp_outs(model, {LAYER: shakespeare_w}):
        lg_ed, coh_e, wins_e, col_e = mg.gain_sweep(model, tok, ids, cfg.template, batch=64, return_window_coherence=True)
        per_layer_e = [{"layer": l, "score": 0.0, "median_log_gain": float(lg_ed[l].median()), "top": []} for l in range(12)]
        # With high kl_min so causal_probe doesn't short-circuit, _check_memit_block must reach _swapped_mlp_outs(model, rest_w)
        cfg_strict = mg.MLPGainConfig(tokens="proper", max_tokens=256, kl_min=999.0)
        edits_strict, _ = mg._check_memit_block(
            model, tok, lg_ed, ids, valid_cand, ref, per_layer_e, templates, cfg_strict,
            coh_win=coh_e, wins=wins_e, col_lg=col_e, content_mask=cmask,
        )
        assert edits_strict == []


def test_no_undefined_names_in_package():
    """Static AST check ensuring no undefined global/local names exist in llm_forensics."""
    import ast
    import builtins
    import pathlib

    builtin_names = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__all__", "annotations"}
    pkg_root = pathlib.Path(mg.__file__).resolve().parent.parent

    def check_file(path: pathlib.Path) -> list[tuple[int, str]]:
        tree = ast.parse(path.read_text())
        errors: list[tuple[int, str]] = []

        def collect_targets(target, out: set[str]):
            if isinstance(target, ast.Name):
                out.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    collect_targets(elt, out)
            elif isinstance(target, ast.Starred):
                collect_targets(target.value, out)

        def collect_scope_defs(nodes) -> set[str]:
            defs: set[str] = set()
            for node in nodes:
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        defs.add(sub.name)
                    if sub is not node and isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                        continue
                    if isinstance(sub, ast.Import):
                        for alias in sub.names:
                            defs.add((alias.asname or alias.name).split(".")[0])
                    elif isinstance(sub, ast.ImportFrom):
                        for alias in sub.names:
                            if alias.name != "*":
                                defs.add(alias.asname or alias.name)
                    elif isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            collect_targets(t, defs)
                    elif isinstance(sub, ast.AnnAssign) and sub.target:
                        collect_targets(sub.target, defs)
                    elif isinstance(sub, ast.AugAssign):
                        collect_targets(sub.target, defs)
                    elif isinstance(sub, ast.NamedExpr):
                        collect_targets(sub.target, defs)
                    elif isinstance(sub, (ast.For, ast.AsyncFor)):
                        collect_targets(sub.target, defs)
                    elif isinstance(sub, (ast.With, ast.AsyncWith)):
                        for item in sub.items:
                            if item.optional_vars:
                                collect_targets(item.optional_vars, defs)
                    elif isinstance(sub, ast.ExceptHandler) and sub.name:
                        defs.add(sub.name)
                    elif isinstance(sub, ast.MatchAs) and sub.name:
                        defs.add(sub.name)
            return defs

        def get_args(args: ast.arguments) -> set[str]:
            out: set[str] = set()
            for a in args.posonlyargs + args.args + args.kwonlyargs:
                out.add(a.arg)
            if args.vararg:
                out.add(args.vararg.arg)
            if args.kwarg:
                out.add(args.kwarg.arg)
            return out

        def walk_scope(nodes, visible: set[str]):
            local_defs = collect_scope_defs(nodes)
            cur_vis = visible | local_defs
            for node in nodes:
                walk_node(node, cur_vis)

        def walk_node(node, visible: set[str]):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    walk_node(dec, visible)
                for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                    walk_node(d, visible)
                fn_vis = visible | {node.name} | get_args(node.args)
                walk_scope(node.body, fn_vis)
            elif isinstance(node, ast.Lambda):
                for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                    walk_node(d, visible)
                fn_vis = visible | get_args(node.args)
                walk_node(node.body, fn_vis)
            elif isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    walk_node(dec, visible)
                for b in node.bases:
                    walk_node(b, visible)
                cls_defs = collect_scope_defs(node.body)
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        walk_node(stmt, visible)
                    else:
                        walk_node(stmt, visible | cls_defs)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                comp_vis = set(visible)
                for gen in node.generators:
                    walk_node(gen.iter, comp_vis)
                    t_defs: set[str] = set()
                    collect_targets(gen.target, t_defs)
                    comp_vis |= t_defs
                    for if_clause in gen.ifs:
                        walk_node(if_clause, comp_vis)
                if isinstance(node, ast.DictComp):
                    walk_node(node.key, comp_vis)
                    walk_node(node.value, comp_vis)
                else:
                    walk_node(node.elt, comp_vis)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in visible and node.id not in builtin_names:
                    errors.append((node.lineno, node.id))
            else:
                for child in ast.iter_child_nodes(node):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and child is node.args:
                        continue
                    walk_node(child, visible)

        walk_scope(tree.body, set())
        return errors

    missing = {}
    for p in sorted(pkg_root.rglob("*.py")):
        errs = check_file(p)
        if errs:
            missing[str(p.relative_to(pkg_root))] = errs
    assert missing == {}, f"Undefined names found: {missing}"
