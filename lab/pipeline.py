"""Research pipeline for one edit config: edit, scan, score, write summary.json.

    python -m lab run configs/edits/x.yaml          # make all CONFIG=...

Files for an experiment named <name> (the config's `name:` or its file stem):

    runs/<name>/model/                      edited weights (git-ignored, large)
    runs/<name>/edit_manifest.json          what `llmf edit` planted; only scoring reads it
    runs/<name>/edit_eval.json              efficacy, paraphrase and neighborhood scores
    results/<name>/scan_with_base.json      `llmf scan --base` report (weight-diff, LLM investigators with the base)
    results/<name>/scan_weights_only.json   `llmf scan` report (spectral, rank-reversal, LLM investigators)
    results/<name>/<detector>.json          each detector's ForensicReport, split out of the scan reports
    results/<name>/summary.json             everything the site needs (committed to git)

Detection is done by the end-user scanner (`llmf scan --json`); this module runs the steps
and scores the results with lab/scoring.py. Each step is a subprocess, so a crash in one
doesn't stop the others, and this module never imports torch.
"""
from __future__ import annotations

import datetime as _dt
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

from lab import scoring
from llm_forensics.forensics.report_schema import validate

ROOT = Path(__file__).resolve().parent.parent  # repo root (lab/ lives next to llm_forensics/)

ALL_STEPS = ["edit", "scan_with_base", "scan_weights_only"]
SCAN_FILES = {"scan_with_base": "with-base", "scan_weights_only": "weights-only"}
CONDITIONS = {"with-base": "with base", "weights-only": "weights only"}


# --------------------------------------------------------------------------- helpers

def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from the repo's .env; never overrides variables already set."""
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
        for line in proc.stdout:          # tee to terminal and log file
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def has_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.call(
        ["nvidia-smi", "-L"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- experiment

@dataclass
class Experiment:
    config_path: Path
    name: str
    base: str | None
    cfg: dict[str, Any] = field(default_factory=dict)
    device: str | None = None          # default: $LLMF_DEVICE, else cuda if nvidia-smi works, else cpu
    llm: str | None = None             # --llm for the scans (default: every provider with a key)

    @property
    def run_dir(self) -> Path:
        return ROOT / "runs" / self.name

    @property
    def results(self) -> Path:
        return ROOT / "results" / self.name

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "edit_manifest.json"

    def has_edit(self) -> bool:
        return (self.run_dir / "model").exists() or (self.run_dir / "delta.safetensors").exists()

    @classmethod
    def from_config(cls, path: str | Path, name: str | None = None, **kw) -> "Experiment":
        import yaml  # only needed when running experiments
        p = Path(path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        cfg = yaml.safe_load(p.read_text()) or {}
        name = name or cfg.get("name") or p.stem
        return cls(config_path=p, name=str(name), base=cfg.get("base") or cfg.get("model"), cfg=cfg, **kw)

    def command(self, step: str) -> list[str]:
        llmf = [sys.executable, "-m", "llm_forensics.cli"]
        device = self.device or os.environ.get("LLMF_DEVICE") or ("cuda" if has_gpu() else "cpu")
        if step == "edit":
            return llmf + ["edit", str(self.config_path), "--out", str(self.run_dir), "--device", device]
        cmd = llmf + ["scan", str(self.run_dir), "--device", device, "--plain", "--no-banner", "--exit-zero",
                      "--full-reports", "--width", "140", "--json", str(self.output(step))]
        if step == "scan_with_base":
            base = self.base or (read_json(self.manifest_path) or {}).get("base_model")
            cmd += ["--base", str(base or "MISSING_BASE_MODEL"), "--detectors", "weight-diff,llm"]
        else:
            cmd += ["--detectors", "spectral,rank-reversal,llm"]
        if self.llm:
            cmd += ["--llm", self.llm]
        return cmd

    def output(self, step: str) -> Path:
        return self.manifest_path if step == "edit" else self.results / f"{step}.json"


# --------------------------------------------------------------------------- running

def run(exp: Experiment, steps: list[str], *, force: bool = False, dry_run: bool = False,
        keep_going: bool = False, summarize_after: bool = True) -> int:
    load_dotenv()
    if not dry_run:
        exp.results.mkdir(parents=True, exist_ok=True)
    log_path = exp.results / "pipeline_log.json"
    log = read_json(log_path) or {"steps": {}}
    failed = 0
    for step in steps:
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


def detector_key(det: dict[str, Any], scan_label: str) -> str:
    """Summary key for a detector entry of a scan report. LLM investigators run in both scans,
    so their key carries the condition (llm-anthropic-with-base, llm-anthropic-weights-only)."""
    return f"{det['id']}-{scan_label}" if det.get("group") == "llm" else str(det["id"])


def scan_detectors(results_dir: Path) -> list[tuple[str, str, dict[str, Any]]]:
    """[(detector key, scan label, detector entry)] from results/<name>/scan_*.json, skipping SKIPPED ones."""
    out = []
    for fname, label in SCAN_FILES.items():
        scan = read_json(results_dir / f"{fname}.json")
        if not isinstance(scan, dict):
            continue
        for det in scan.get("detectors") or []:
            if det.get("status") != "SKIPPED":
                out.append((detector_key(det, label), label, det))
    return out


def split_scan_reports(results_dir: Path) -> list[str]:
    """Write each detector's ForensicReport from the scan JSONs to results/<name>/<key>.json."""
    written = []
    for key, _, det in scan_detectors(results_dir):
        if isinstance(det.get("report"), dict):
            (results_dir / f"{key}.json").write_text(json.dumps(det["report"], indent=2, default=str))
            written.append(key)
    return written


def summarize_dir(results_dir: Path, manifest: dict[str, Any] | None = None, edit_eval: Any = None,
                  meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build summary.json content from a results directory (also used by lab/site/build.py)."""
    detectors: dict[str, Any] = {}
    for key, label, det in scan_detectors(results_dir):
        rep = det.get("report") if isinstance(det.get("report"), dict) else None
        d: dict[str, Any] = {"file": f"{key}.json", "id": det.get("id"), "title": det.get("title") or det.get("id"),
                             "condition": CONDITIONS[label] + (", raw weights" if (det.get("metrics") or {}).get(
                                 "llm_mode") == "raw" else ""), "status": det.get("status"),
                             "metrics": det.get("metrics") or {}}
        if rep is None:  # the detector errored inside the scan
            detectors[key] = {**d, "error": det.get("error"), "detected": None, "confidence": None}
            continue
        d.update({"detected": rep.get("detected"), "confidence": rep.get("confidence"),
                  "runtime_s": rep.get("runtime_s"), "n_suspected": len(rep.get("suspected_edits") or []),
                  "top": _top(rep), "schema_problems": validate(rep)})
        for k in ("investigator_model", "stop_reason"):
            if k in rep:
                d[k] = rep[k]
        if "cost_usd" in (rep.get("usage") or {}):
            d["cost_usd"] = rep["usage"]["cost_usd"]
        if manifest is not None:
            try:
                d["eval"] = scoring.evaluate(rep, manifest)
            except Exception as e:  # noqa: BLE001 - one bad report must not stop the summary
                d["eval_error"] = f"{type(e).__name__}: {e}"
        detectors[key] = d
    return {
        "schema": "llmf-summary/3",
        **(meta or {}),
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "expected_edits": scoring.expected_edits(manifest) if manifest is not None else None,
        "edit_manifest": manifest,
        "edit_eval": edit_eval,
        "detectors": detectors,
        "pipeline": read_json(results_dir / "pipeline_log.json"),
    }


def summarize(exp: Experiment) -> dict[str, Any]:
    previous = read_json(exp.results / "summary.json") or {}
    # results/ is committed without runs/, so fall back to the copies kept in the last summary
    manifest = read_json(exp.manifest_path) or previous.get("edit_manifest")
    if manifest is None:
        print(f"!! no edit manifest for {exp.name} ({exp.manifest_path.relative_to(ROOT)}); "
              "the summary will have no scores")
    edit_eval = read_json(exp.run_dir / "edit_eval.json") or previous.get("edit_eval")
    split_scan_reports(exp.results)
    meta = {"name": exp.name, "base_model": exp.base,
            "config_path": str(exp.config_path.relative_to(ROOT)) if exp.config_path.is_relative_to(ROOT)
            else str(exp.config_path), "config": exp.cfg}
    summary = summarize_dir(exp.results, manifest, edit_eval, meta)
    exp.results.mkdir(parents=True, exist_ok=True)
    path = exp.results / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {path.relative_to(ROOT)}")
    print_table(summary)
    return summary


def fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 10 ** -nd or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.{nd}f}"
    return str(v)


def print_table(summary: dict[str, Any]) -> None:
    cols = ["detected", "confidence", "layer_correct", "subject_hit@1", "subject_hit@10", "target_hit@1",
            "target_hit@10"]
    print(f"\n{'detector':28s}" + "".join(f"{c:>15s}" for c in cols))
    for key, d in summary["detectors"].items():
        ev = d.get("eval") or {}
        vals = [d.get("detected"), d.get("confidence")] + [ev.get(c) for c in cols[2:]]
        print(f"{key:28s}" + "".join(f"{fmt(v):>15s}" for v in vals))


# --------------------------------------------------------------------------- environment check

def check_env() -> int:
    """Print an environment checklist; return non-zero if a required check fails."""
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
        line(False, "torch import failed", "run `make setup`")
    else:
        info = json.loads(r.stdout.strip().splitlines()[-1])
        line(True, f"torch {info['torch']}")
        if info["cuda"]:
            line(True, "GPU: " + ", ".join(f"{g} ({m} GiB)" for g, m in zip(info["gpus"], info["mem"])))
        else:
            line(None, "no CUDA GPU: fine for gpt2 and the exercises, slow for 7B models",
                 "see 'GPU runs on SaladCloud' in lab/README.md (make salad-run)")

    hf = os.environ.get("HF_TOKEN") or (Path.home() / ".cache" / "huggingface" / "token").exists()
    line(True if hf else None, "Hugging Face token " + ("found" if hf else "not found"),
         "" if hf else "only needed for gated models (Llama): `hf auth login` or HF_TOKEN in .env")
    from llm_forensics.scan.context import PROVIDER_ENV, providers_from_env  # same lookup as `llmf scan`
    for prov, src in providers_from_env().items():
        line(True if src else None, f"llm-{prov} detector: " + (f"key from {src}" if src else "no key"),
             "" if src else f"set {' or '.join(PROVIDER_ENV[prov])} in .env or ~/.config/llm_forensics/{prov}.env")
    free = shutil.disk_usage(ROOT).free / 2**30
    line(free > 30 or None, f"{free:.0f} GiB free disk", "" if free > 30 else "7B models need ~15 GiB each")
    print("\nAll required checks passed." if ok else "\nSome required checks failed.")
    return 0 if ok else 1
