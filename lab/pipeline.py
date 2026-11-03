"""Research pipeline for one config: edit -> scan -> evaluate vs ground truth -> summary.json.

    python -m lab run configs/edits/x.yaml          # (make all CONFIG=...)

Layout for an experiment named <name> (config `name:` or the config file stem):

    runs/<name>/model/                  edited weights (git-ignored, large)
    runs/<name>/ground_truth.json       answer key  -- only evaluation reads this
    runs/<name>/edit_eval.json          efficacy / paraphrase / neighbourhood scores
    results/<name>/scan_with_base.json      `llmf scan --base` report (weight-diff, LLM with base)
    results/<name>/scan_weights_only.json   `llmf scan` report (spectral, rank-reversal, LLM weights only)
    results/<name>/<method>.json        one ForensicReport per detector, split out of the scan
                                        reports (shared schema, llm_forensics/forensics/report_schema.py)
    results/<name>/summary.json         everything the site needs (committed to git)

Per-detector results come from the end-user scanner (`llmf scan --json`); this module
only orchestrates and scores.  Each step is a subprocess so a crash never takes down the
others, and this module never imports torch.  A config can tune the steps:

    pipeline:
      device: cuda                       # passed to steps that accept --device
      skip: [scan_with_base]             # steps to leave out
      llm: claude,openai                 # LLM investigators (default: every provider with a key)
      extra_args:
        scan_weights_only: ["--thorough"]
      commands:                          # full override; {placeholders} are filled in
        scan_with_base: "{llmf} scan {suspect} --base {base} --json {results}/scan_with_base.json"

Placeholders: {python} {llmf} {config} {name} {base} {suspect} {run_dir} {results} {device}.
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent  # repo root (lab/ lives next to llm_forensics/)

ALL_STEPS = ["edit", "scan_with_base", "scan_weights_only"]

_SCAN = ("{llmf} scan {suspect} --device {device} --plain --no-banner --exit-zero --full-reports --width 140 "
         "--json {results}/%s.json")
# step -> command template
DEFAULT_COMMANDS: dict[str, str] = {
    "edit": "{llmf} edit {config} --out {run_dir} --device {device}",
    "scan_with_base": _SCAN % "scan_with_base" + " --base {base} --detectors weight-diff,llm",
    "scan_weights_only": _SCAN % "scan_weights_only" + " --detectors spectral,rank-reversal,llm",
}
OUTPUTS: dict[str, str] = {
    "edit": "{run_dir}/ground_truth.json",
    "scan_with_base": "{results}/scan_with_base.json",
    "scan_weights_only": "{results}/scan_weights_only.json",
}
SCAN_FILES = {"scan_with_base": "with-base", "scan_weights_only": "weights-only"}
# per-method files written by older versions of the pipeline (still summarised if present)
LEGACY_METHOD_FILES = ["diff", "blind", "literature", "investigator_with_original", "investigator_blind"]


# --------------------------------------------------------------------------- helpers

def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip("'\"")
        if k and v and k not in os.environ:
            os.environ[k] = v


def sh(cmd: list[str], log_path: Path | None = None, dry_run: bool = False) -> int:
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry_run:
        return 0
    if log_path is None:
        return subprocess.call(cmd, cwd=ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:          # tee to terminal + log file
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.call(
        ["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_report_schema():
    """Import report_schema by path (stdlib-only module) so evaluation works without torch."""
    path = ROOT / "llm_forensics" / "forensics" / "report_schema.py"
    spec = importlib.util.spec_from_file_location("_llmf_report_schema", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod        # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- experiment

@dataclass
class Experiment:
    config_path: Path
    name: str
    base: str | None
    cfg: dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return ROOT / "runs" / self.name

    @property
    def results(self) -> Path:
        return ROOT / "results" / self.name

    @property
    def suspect(self) -> Path:
        """Edited weights: run_dir/model (full save) or the run dir itself (delta-only save)."""
        return self.run_dir / "model" if (self.run_dir / "model").exists() else self.run_dir

    def has_edit(self) -> bool:
        return (self.run_dir / "model").exists() or (self.run_dir / "delta.safetensors").exists()

    @property
    def pipe(self) -> dict[str, Any]:
        return self.cfg.get("pipeline") or {}

    @classmethod
    def from_config(cls, path: str | Path, name: str | None = None) -> "Experiment":
        import yaml  # lazy: only needed when running experiments
        p = Path(path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        cfg = yaml.safe_load(p.read_text()) or {}
        name = name or cfg.get("name") or p.stem
        return cls(config_path=p, name=str(name), base=find_base_model(cfg), cfg=cfg)

    def placeholders(self) -> dict[str, str]:
        device = str(self.pipe.get("device") or os.environ.get("LLMF_DEVICE") or ("cuda" if has_gpu() else "cpu"))
        base = self.base
        if base is None:  # fall back to whatever the editor recorded
            gt = _read_json(self.run_dir / "ground_truth.json") or {}
            base = find_base_model(gt) if isinstance(gt, dict) else None
        return {"python": sys.executable, "llmf": f"{sys.executable} -m llm_forensics.cli",
                "config": str(self.config_path), "name": self.name,
                "base": str(base or "MISSING_BASE_MODEL"), "suspect": str(self.suspect),
                "run_dir": str(self.run_dir), "results": str(self.results), "device": device}

    def command(self, step: str) -> list[str]:
        tmpl = (self.pipe.get("commands") or {}).get(step, DEFAULT_COMMANDS[step])
        cmd = shlex.split(tmpl.format(**self.placeholders()))
        cmd += [str(x) for x in (self.pipe.get("extra_args") or {}).get(step, [])]
        if step.startswith("scan") and step not in (self.pipe.get("commands") or {}):
            if self.pipe.get("llm"):
                llm = self.pipe["llm"]
                cmd += ["--llm", llm if isinstance(llm, str) else ",".join(llm)]
        return cmd

    def output(self, step: str) -> Path:
        return Path(OUTPUTS[step].format(run_dir=self.run_dir, results=self.results))


def find_base_model(cfg: dict[str, Any]) -> str | None:
    for k in ("base_model", "model", "model_name", "base"):
        v = cfg.get(k)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            for kk in ("name", "id", "path", "name_or_path"):
                if isinstance(v.get(kk), str):
                    return v[kk]
    return None


# --------------------------------------------------------------------------- running

def run(exp: Experiment, steps: list[str], *, force: bool = False, dry_run: bool = False,
        keep_going: bool = False, summarize_after: bool = True) -> int:
    load_dotenv()
    skip_cfg = set(exp.pipe.get("skip") or [])
    if not dry_run:
        exp.results.mkdir(parents=True, exist_ok=True)
    log_path = exp.results / "pipeline_log.json"
    log = _read_json(log_path) or {"steps": {}}
    failed = 0
    for step in steps:
        if step in skip_cfg:
            print(f"== {step}: skipped by config")
            continue
        out = exp.output(step)
        if out.exists() and not force:
            print(f"== {step}: {out.relative_to(ROOT)} exists, skipping (use --force to redo)")
            continue
        if step != "edit" and not exp.has_edit() and not dry_run:
            print(f"== {step}: no edited model in {exp.run_dir.relative_to(ROOT)}; run the edit step first")
            log["steps"][step] = {"status": "skipped", "reason": "no edited model"}
            failed += 1
            if not keep_going:
                break
            continue
        print(f"\n== {step} " + "=" * 60)
        t0 = time.time()
        rc = sh(exp.command(step), log_path=exp.results / "logs" / f"{step}.log", dry_run=dry_run)
        status = "ok" if rc == 0 else "failed"
        log["steps"][step] = {"status": status, "returncode": rc, "seconds": round(time.time() - t0, 1),
                              "finished": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
        if not dry_run:
            log_path.write_text(json.dumps(log, indent=2))
        if rc != 0:
            failed += 1
            print(f"!! {step} failed (rc={rc}); log: results/{exp.name}/logs/{step}.log")
            if step == "edit" or not keep_going:
                break
    if summarize_after and not dry_run:
        summarize(exp)
    return 1 if failed else 0


# --------------------------------------------------------------------------- evaluation

def _top(report: dict[str, Any]) -> dict[str, Any]:
    eds = report.get("suspected_edits") or []
    if not eds:
        return {}
    e = eds[0]
    sub = (e.get("subject_candidates") or [{}])[0]
    tgt = (e.get("target_candidates") or [{}])[0]
    return {"layer": e.get("layer"), "tensor": e.get("tensor"), "rank": e.get("rank"),
            "subject": sub.get("text"), "target": tgt.get("token")}


def method_key(det: dict[str, Any], scan_label: str) -> str:
    """Site/summary key for a detector entry of a scan report: LLM investigators run in both
    scans, so their key carries the mode (llm-anthropic-with-base / -weights-only)."""
    return f"{det['id']}-{scan_label}" if det.get("group") == "llm" else str(det["id"])


def scan_methods(results_dir: Path) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """[(method key, detector entry, scan report)] from results/<name>/scan_*.json."""
    out = []
    for fname, label in SCAN_FILES.items():
        scan = _read_json(results_dir / f"{fname}.json")
        if not isinstance(scan, dict):
            continue
        for det in scan.get("detectors") or []:
            if det.get("status") == "SKIPPED":
                continue
            out.append((method_key(det, label), det, scan))
    return out


def split_scan_reports(results_dir: Path) -> list[str]:
    """Write each detector's ForensicReport from the scan JSONs to results/<name>/<key>.json."""
    written = []
    for key, det, _ in scan_methods(results_dir):
        if isinstance(det.get("report"), dict):
            (results_dir / f"{key}.json").write_text(json.dumps(det["report"], indent=2, default=str))
            written.append(key)
    return written


def summarize_dir(results_dir: Path, ground_truth: Any = None, edit_eval: Any = None,
                  meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build summary.json content from a results directory (also used by lab/site/build.py)."""
    rs = _load_report_schema()
    entries: list[tuple[str, dict[str, Any] | None, dict[str, Any]]] = []
    for key, det, _ in scan_methods(results_dir):
        rep = det.get("report") if isinstance(det.get("report"), dict) else None
        entries.append((key, rep, {"file": f"{key}.json", "status": det.get("status"), "error": det.get("error"),
                                   "metrics": det.get("metrics") or {}}))
    for key in LEGACY_METHOD_FILES:
        if any(k == key for k, _, _ in entries):
            continue
        rep = _read_json(results_dir / f"{key}.json")
        if isinstance(rep, dict):
            entries.append((key, rep, {"file": f"{key}.json", "status": "OK"}))
    methods: dict[str, Any] = {}
    for key, rep, info in entries:
        if rep is None:  # detector errored inside the scan
            methods[key] = {"file": info["file"], "method": key, "status": info.get("status"),
                            "error": info.get("error"), "detected": None, "confidence": None}
            continue
        m: dict[str, Any] = {
            "file": info["file"], "method": rep.get("method", key), "status": info.get("status"),
            "detected": rep.get("detected"), "confidence": rep.get("confidence"),
            "runtime_s": rep.get("runtime_s"), "n_suspected": len(rep.get("suspected_edits") or []),
            "top": _top(rep), "schema_problems": rs.validate(rep),
        }
        for k in ("cost_usd_estimate", "tool_calls", "investigator_model", "stop_reason"):
            if k in rep:
                m[k] = rep[k]
        if ground_truth is not None:
            try:
                m["eval"] = rs.evaluate(rep, ground_truth)
            except Exception as e:  # noqa: BLE001 - never let one bad report kill the summary
                m["eval_error"] = f"{type(e).__name__}: {e}"
        methods[key] = m
    out = {
        "schema": "llmf-summary/2",
        **(meta or {}),
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "ground_truth": rs.normalize_ground_truth(ground_truth) if ground_truth is not None else None,
        "ground_truth_raw": ground_truth,
        "edit_eval": edit_eval,
        "methods": methods,
        "pipeline": _read_json(results_dir / "pipeline_log.json"),
    }
    return out


def summarize(exp: Experiment) -> dict[str, Any] | None:
    gt = _read_json(exp.run_dir / "ground_truth.json")
    if gt is None:
        prev = _read_json(exp.results / "summary.json") or {}
        gt = prev.get("ground_truth_raw")      # results committed without runs/ (e.g. on a laptop)
    if gt is None:
        print(f"!! no ground truth for {exp.name} (runs/{exp.name}/ground_truth.json); "
              "summary will have no scores")
    split_scan_reports(exp.results)
    edit_eval = _read_json(exp.run_dir / "edit_eval.json")
    if edit_eval is None:
        edit_eval = (_read_json(exp.results / "summary.json") or {}).get("edit_eval")
    meta = {"name": exp.name, "base_model": exp.base,
            "config_path": str(exp.config_path.relative_to(ROOT)) if exp.config_path.is_relative_to(ROOT)
            else str(exp.config_path), "config": exp.cfg}
    summary = summarize_dir(exp.results, gt, edit_eval, meta)
    exp.results.mkdir(parents=True, exist_ok=True)
    path = exp.results / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {path.relative_to(ROOT)}")
    _print_table(summary)
    return summary


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _print_table(summary: dict[str, Any]) -> None:
    cols = ["detected", "confidence", "layer_correct", "subject_hit@1", "subject_hit@10", "target_hit@1",
            "target_hit@10"]
    print(f"\n{'method':28s}" + "".join(f"{c:>15s}" for c in cols))
    for key, m in summary["methods"].items():
        ev = m.get("eval") or {}
        vals = [m.get("detected"), m.get("confidence")] + [ev.get(c) for c in cols[2:]]
        print(f"{key:28s}" + "".join(f"{_fmt(v):>15s}" for v in vals))


# --------------------------------------------------------------------------- env check

def check_env() -> int:
    """aisb "day 0"-style setup check. Prints a checklist, returns non-zero on hard failures."""
    load_dotenv()
    ok = True

    def line(good: bool | None, msg: str, hint: str = "") -> None:
        mark = {True: "[ ok ]", False: "[FAIL]", None: "[warn]"}[good]
        print(f"{mark} {msg}" + (f"\n       -> {hint}" if hint else ""))

    v = sys.version_info
    good = v >= (3, 10)
    ok &= good
    line(good, f"python {v.major}.{v.minor}.{v.micro}", "" if good else "need >= 3.10 (uv python install 3.11)")

    probe = ("import json,torch;print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),"
             "'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
             "'mem':[round(torch.cuda.get_device_properties(i).total_memory/2**30,1) "
             "for i in range(torch.cuda.device_count())]}))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if r.returncode != 0:
        ok = False
        line(False, "torch import failed", "run `make setup` (uv sync)")
    else:
        info = json.loads(r.stdout.strip().splitlines()[-1])
        line(True, f"torch {info['torch']}")
        if info["cuda"]:
            line(True, "GPU: " + ", ".join(f"{g} ({m} GiB)" for g, m in zip(info["gpus"], info["mem"])))
        else:
            line(None, "no CUDA GPU - fine for gpt2 / exercises, slow for 7B models",
                 "see README 'GPU runs on SaladCloud' (make salad-run)")

    hf = os.environ.get("HF_TOKEN") or (Path.home() / ".cache" / "huggingface" / "token").exists()
    line(True if hf else None, "Hugging Face token " + ("found" if hf else "not found"),
         "" if hf else "only needed for gated models (Llama): `hf auth login` or HF_TOKEN in .env")
    for prov, vars_ in (("anthropic", ("ANTHROPIC_API_KEY",)), ("openai", ("OPENAI_API_KEY",)),
                        ("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"))):
        have = next((v for v in vars_ if os.environ.get(v)), None)
        line(True if have else None, f"{prov} LLM investigator: " + (f"{have} set" if have else "no key"),
             "" if have else f"set {' or '.join(vars_)} in .env to add the llm-{prov} detector")
    free = shutil.disk_usage(ROOT).free / 2**30
    line(free > 30 or None, f"{free:.0f} GiB free disk", "" if free > 30 else "7B models need ~15 GiB each")
    print("\nAll required checks passed." if ok else "\nSome required checks failed.")
    return 0 if ok else 1
