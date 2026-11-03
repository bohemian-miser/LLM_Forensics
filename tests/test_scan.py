"""Tests for `llmf scan`: registry / discovery, findings, rendering, JSON, and an end-to-end
scan of gpt2 with a synthetic rank-one edit (weight-diff + spectral, --fast).

Run: .venv/bin/python -m pytest -q tests/test_scan.py
"""
from __future__ import annotations

import io
import json

import pytest

from llm_forensics import cli as top_cli
from llm_forensics.scan import registry
from llm_forensics.scan.cli import choose_detectors
from llm_forensics.scan.context import ScanContext, ScanOptions, providers_from_env
from llm_forensics.scan.findings import Finding, Severity, from_report, severity_for
from llm_forensics.scan.registry import Detector, DetectorResult, register_detector, run_detector, select
from llm_forensics.scan import render
from llm_forensics.scan.runner import run_scan

ROME_REPORT = {
    "method": "x", "model": "m", "detected": True, "confidence": 0.97, "notes": "", "runtime_s": 1.0,
    "suspected_edits": [{
        "layer": 5, "tensor": "transformer.h.5.mlp.c_proj.weight", "rank": 1, "delta_norm": 6.8,
        "subject_candidates": [{"text": "Eiffel Tower", "score": 7.2}, {"text": "CN Tower", "score": 6.0}],
        "target_candidates": [{"token": " Rome", "score": 14.7}],
        "evidence": {"rel_norm": 0.02}}],
}


# --------------------------------------------------------------------------- dummy detectors

class DummyEdit(Detector):
    name = "dummy-edit"
    title = "Dummy edit finder"
    edit_types = frozenset({"rome"})
    default = False

    def run(self, ctx):
        return DetectorResult(findings=from_report(ROME_REPORT, self.id, prefix="DUMMY"), detected=True,
                              confidence=0.97, layer=5, layer_scores={l: (1.0 if l == 5 else 0.1) for l in range(12)},
                              report=ROME_REPORT)


class DummyClean(Detector):
    name = "dummy-clean"
    default = False

    def run(self, ctx):
        return DetectorResult(detected=False, confidence=0.2, layer_scores={l: 0.05 * l for l in range(12)})


class DummyBoom(Detector):
    name = "dummy-boom"
    default = False

    def run(self, ctx):
        raise RuntimeError("kaboom")


class DummyNeedsBase(Detector):
    name = "dummy-needs-base"
    requires_base = True
    default = False

    def run(self, ctx):  # pragma: no cover - never reached without a base
        return DetectorResult()


class FakeLLMDetector(Detector):
    name = "llm"
    group = "llm"
    default = False

    @property
    def id(self):
        return "llm-fake"

    def run(self, ctx):
        return DetectorResult(findings=from_report(ROME_REPORT, self.id), detected=True, confidence=0.9, layer=5,
                              metrics={"provider": "fake", "model": "fake-1", "tool_calls": 7, "tokens": 12345,
                                       "cost_usd": 0.0123, "wall_s": 42.0})


@pytest.fixture
def dummies():
    classes = [DummyEdit, DummyClean, DummyBoom, DummyNeedsBase]
    for c in classes:
        register_detector(c)
    yield [c() for c in classes]
    for c in classes:
        registry.unregister_detector(c.name)


def _ctx():
    return ScanContext("not-loaded-model", options=ScanOptions(profile="fast"))


def _report(dets):
    return run_scan(_ctx(), dets, target={"name": "toy", "arch": "gpt2", "layers": 12, "params": 124_000_000,
                                          "dtype": "float32", "device": "cpu"})


# --------------------------------------------------------------------------- registry

def test_builtins_discovered():
    ids = {d.id for d in registry.all_detectors()}
    assert {"weight-diff", "spectral", "rank-reversal"} <= ids
    assert "llm" in registry.discover()
    wd = next(d for d in registry.all_detectors() if d.id == "weight-diff")
    assert wd.requires_base and "rome" in wd.edit_types


def test_register_and_select(dummies):
    assert "dummy-edit" in registry.discover()
    dets = registry.all_detectors() + [FakeLLMDetector()]
    assert [d.id for d in select(["dummy-edit"], dets)] == ["dummy-edit"]
    assert all(d.group == "llm" for d in select(["llm"], dets))
    assert "dummy-edit" not in {d.id for d in select(None, dets)}  # default=False
    with pytest.raises(ValueError, match="unknown detector"):
        select(["nope"], dets)
    with pytest.raises(TypeError):
        register_detector(object)  # type: ignore[arg-type]


def test_entry_point_plugins(monkeypatch):
    class Plug(Detector):
        name = "plugin-x"

        def run(self, ctx):
            return DetectorResult()

    class EP:
        def __init__(self, name, obj):
            self.name, self._obj = name, obj

        def load(self):
            if isinstance(self._obj, Exception):
                raise self._obj
            return self._obj

    import importlib.metadata as md
    monkeypatch.setattr(md, "entry_points", lambda group: [EP("good", Plug), EP("bad", ImportError("no module"))])
    monkeypatch.setattr(registry, "_DISCOVERED", False)
    monkeypatch.setattr(registry, "DISCOVERY_ERRORS", [])
    try:
        assert "plugin-x" in registry.discover()
        assert any("bad" in e for e in registry.DISCOVERY_ERRORS)
    finally:
        registry.unregister_detector("plugin-x")


def test_default_selection_rules(dummies):
    no_base = {d.id for d in choose_detectors(None, None, has_base=False)}
    assert "weight-diff" not in no_base and {"spectral", "rank-reversal"} <= no_base
    with_base = {d.id for d in choose_detectors(None, None, has_base=True, no_llm=True)}
    assert "weight-diff" in with_base and not any(i.startswith("llm-") for i in with_base)
    openai = {d.id for d in choose_detectors(["spectral"], ["openai"], has_base=False)}
    assert openai == {"spectral", "llm-openai"}


def test_run_detector_errors_and_skips(dummies):
    ctx = _ctx()
    boom = run_detector(DummyBoom(), ctx)
    assert boom.status == "ERROR" and "kaboom" in boom.error
    skip = run_detector(DummyNeedsBase(), ctx)
    assert skip.status == "SKIPPED" and "--base" in skip.error


def test_provider_keys_never_expose_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-secret")
    keys = providers_from_env(key_files=False)
    assert keys["openai"] == "OPENAI_API_KEY" and keys["gemini"] == "GOOGLE_API_KEY"
    assert "secret" not in json.dumps(keys)


# --------------------------------------------------------------------------- findings

def test_severity_rules():
    assert severity_for(0.95, True, True) == Severity.CRITICAL
    assert severity_for(0.95, False, True) == Severity.HIGH
    assert severity_for(0.95, True, False) == Severity.MEDIUM
    assert severity_for(0.7, True, True) == Severity.MEDIUM
    assert severity_for(0.4, False, True) == Severity.LOW
    assert severity_for(0.1, False, True) == Severity.INFO


def test_from_report():
    fs = from_report(ROME_REPORT, "weight-diff", prefix="DIFF")
    assert len(fs) == 1
    f = fs[0]
    assert (f.id, f.severity, f.layer, f.edit_type) == ("LLMF-DIFF-001", Severity.CRITICAL, 5, "rome")
    assert f.subject == "Eiffel Tower" and f.target == " Rome"
    assert from_report({**ROME_REPORT, "detected": False}, "x") == []
    memit = {**ROME_REPORT, "suspected_edits": [dict(ROME_REPORT["suspected_edits"][0], layer=l,
                                                     tensor=f"transformer.h.{l}.mlp.c_proj.weight") for l in (3, 4)]}
    assert {f.edit_type for f in from_report(memit, "x")} == {"memit"}


# --------------------------------------------------------------------------- rendering + JSON

def _render(report, **kw) -> str:
    buf = io.StringIO()
    con = render.make_console(file=buf, width=130, plain=True)
    render.banner(con)
    con.print(render.target_panel(report.target, report.detectors, {"anthropic": "ANTHROPIC_API_KEY",
                                                                     "openai": None, "gemini": None}, "fast"))
    render.render_results(con, report, **kw)
    return buf.getvalue()


def test_render_report(dummies):
    rep = _report(dummies)
    out = _render(rep, json_path="out.json")
    assert "\x1b[" not in out                      # plain: no ANSI escapes
    for s in ("|_____|", "Scan target", "Findings", "LLMF-DUMMY-001", "CRITICAL",
              "Eiffel Tower", "' Rome'", "Suspected trigger subject", "Suspected injected target",
              "Layer anomalies", "Detector agreement", "Summary", "ERROR", "kaboom", "SKIPPED", "exit code 1",
              "IMPLANT SUSPECTED", "● anthropic", "○ openai"):
        assert s in out, s
    assert "█" in out                              # heat strip
    assert "✘" in out                              # dummy-edit vs dummy-clean disagree
    assert "1/2 detectors flag an edit" in out


def test_banner_fits_80_columns():
    for line in (render.LENS_ART + "\n" + render.TITLE_ART).splitlines():
        assert len(line) <= 80


def test_llm_table_renders():
    rep = _report([FakeLLMDetector()])
    out = _render(rep)
    assert "LLM investigators" in out and "fake-1" in out and "12,345" in out and "0.0123" in out


def test_json_output(dummies):
    rep = _report(dummies)
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["schema"] == "llmf-scan/1"
    assert d["summary"]["counts"]["CRITICAL"] == 1 and d["summary"]["exit_code"] == 1
    assert d["findings"][0]["id"] == "LLMF-DUMMY-001" and d["findings"][0]["severity"] == "CRITICAL"
    st = {x["id"]: x["status"] for x in d["detectors"]}
    assert st == {"dummy-edit": "OK", "dummy-clean": "OK", "dummy-boom": "ERROR", "dummy-needs-base": "SKIPPED"}
    assert d["agreement"]["n_flagging"] == 1 and d["agreement"]["pairwise"]["dummy-edit|dummy-clean"] == "disagree"
    assert d["layer_heat"]["5"] == 1.0


def test_exit_codes(dummies):
    assert _report([DummyClean()]).exit_code == 0
    assert _report([DummyBoom()]).exit_code == 3
    assert _report([DummyEdit(), DummyBoom()]).exit_code == 1


def test_top_level_cli_commands(capsys):
    with pytest.raises(SystemExit):
        top_cli.main(["--help"])
    out = capsys.readouterr().out
    assert "{scan,edit,chat}" in out
    for gone in ("diff", "blind", "literature", "investigate", "evaluate"):
        assert f"    {gone} " not in out


# --------------------------------------------------------------------------- end to end (gpt2)

@pytest.fixture(scope="module")
def suspect_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    from llm_forensics.investigator.fixture import make_synthetic_suspect
    from llm_forensics.model_io import load_model
    torch.manual_seed(0)
    base, tok = load_model("gpt2", "cpu")
    suspect, truth = make_synthetic_suspect(base, tok)
    d = tmp_path_factory.mktemp("suspect")
    suspect.save_pretrained(d)
    tok.save_pretrained(d)
    return d, truth


def test_end_to_end_gpt2(suspect_dir, tmp_path, capsys):
    from llm_forensics.scan.cli import main
    d, truth = suspect_dir
    out_json, out_html = tmp_path / "scan.json", tmp_path / "scan.html"
    rc = main([str(d), "--base", "gpt2", "--detectors", "weight-diff,spectral", "--fast", "--plain",
               "--json", str(out_json), "--html", str(out_html), "--width", "130"])
    text = capsys.readouterr().out
    rep = json.loads(out_json.read_text())
    st = {x["id"]: x for x in rep["detectors"]}
    assert st["weight-diff"]["status"] == "OK", st["weight-diff"]["error"]
    assert st["spectral"]["status"] in ("OK", "ERROR")  # spectral is evaluated in its own tests
    diff = [f for f in rep["findings"] if f["detector"] == "weight-diff" and f["layer"] == truth["layer"]]
    assert diff and diff[0]["severity"] in ("HIGH", "CRITICAL")
    assert truth["target"] in diff[0]["target_candidates"][:5]
    assert any("Eiffel" in s for s in diff[0]["subject_candidates"][:5])
    assert rc == 1 and rep["summary"]["exit_code"] == 1
    assert "weight-diff" in text and "LLMF-DIFF-001" in text and "\x1b[" not in text
    assert out_html.read_text().lstrip().lower().startswith("<!doctype html")
