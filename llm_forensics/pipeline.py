"""Run a full experiment for one config: edit -> forensics -> evaluate -> summary.json.

    python -m llm_forensics.pipeline configs/edits/x.yaml          # == `llmf all ...`

Layout for an experiment named <name> (config `name:` or the config file stem):

    runs/<name>/model/              edited weights (git-ignored, large)
    runs/<name>/ground_truth.json   answer key  -- only evaluation reads this
    runs/<name>/edit_eval.json      efficacy / paraphrase / neighbourhood scores
    results/<name>/diff.json        one report per forensic method (shared schema,
    results/<name>/blind.json         see llm_forensics/forensics/report_schema.py)
    results/<name>/literature.json
    results/<name>/investigator_with_original.json
    results/<name>/investigator_blind.json
    results/<name>/summary.json     everything the site needs (committed to git)

Each step is a subprocess `python -m ...` so a crash in one method never takes down
the others, and this module never imports torch. A config can override the command
for any step and append extra arguments:

    pipeline:
      device: cuda                       # passed to steps that accept --device
      skip: [literature]                 # steps to leave out of `llmf all`
      extra_args:
        investigate_blind: ["--max-tool-calls", "60"]
      commands:                          # full override; {placeholders} are filled in
        diff: "python -m my.diff --a {base} --b {suspect} --out {results}/diff.json"

Placeholders: {python} {config} {name} {base} {suspect} {run_dir} {results} {device}.
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

ROOT = Path(__file__).resolve().parent.parent

ALL_STEPS = ["edit", "diff", "blind", "literature", "investigate_with_original", "investigate_blind"]

# step -> (command template, output file relative to results/<name> or run_dir)
DEFAULT_COMMANDS: dict[str, str] = {
    "edit": "{python} -m llm_forensics.edit.run {config} --out {run_dir} --device {device}",
    "diff": "{python} -m llm_forensics.forensics.diff --base {base} --suspect {suspect} --device {device} "
            "--out {results}/diff.json",
    "blind": "{python} -m llm_forensics.forensics.blind --suspect {suspect} --device {device} --out {results}/blind.json",
    "literature": "{python} -m llm_forensics.forensics.literature --suspect {suspect} --device {device} "
                  "--out {results}/literature.json",
    "investigate_with_original": "{python} -m llm_forensics.investigator.run --suspect {suspect} --base {base} "
                                 "--mode with_original --device {device} --out {results}",
    "investigate_blind": "{python} -m llm_forensics.investigator.run --suspect {suspect} "
                         "--mode blind --device {device} --out {results}",
}
OUTPUTS: dict[str, str] = {
    "edit": "{run_dir}/ground_truth.json",
    "diff": "{results}/diff.json",
    "blind": "{results}/blind.json",
    "literature": "{results}/literature.json",
    "investigate_with_original": "{results}/investigator_with_original.json",
    "investigate_blind": "{results}/investigator_blind.json",
}
METHOD_FILES = ["diff", "blind", "literature", "investigator_with_original", "investigator_blind"]


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
        return self.run_dir / "model"

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
        return {"python": sys.executable, "config": str(self.config_path), "name": self.name,
                "base": str(base or "MISSING_BASE_MODEL"), "suspect": str(self.suspect),
                "run_dir": str(self.run_dir), "results": str(self.results), "device": device}

    def command(self, step: str) -> list[str]:
        tmpl = (self.pipe.get("commands") or {}).get(step, DEFAULT_COMMANDS[step])
        cmd = shlex.split(tmpl.format(**self.placeholders()))
        cmd += [str(x) for x in (self.pipe.get("extra_args") or {}).get(step, [])]
        if step.startswith("investigate") and os.environ.get("LLMF_INVESTIGATOR_DRY_RUN") == "1":
            cmd.append("--dry-run")
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
        if step.startswith("investigate") and not (os.environ.get("ANTHROPIC_API_KEY")
                                                   or os.environ.get("LLMF_INVESTIGATOR_DRY_RUN") == "1"):
            print(f"== {step}: ANTHROPIC_API_KEY not set, skipping "
                  "(set it in .env, or LLMF_INVESTIGATOR_DRY_RUN=1 for the scripted agent)")
            log["steps"][step] = {"status": "skipped", "reason": "no ANTHROPIC_API_KEY"}
            continue
        if step != "edit" and not exp.suspect.exists() and not dry_run:
            print(f"== {step}: no edited model at {exp.suspect.relative_to(ROOT)}; run `llmf edit` first")
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


def summarize_dir(results_dir: Path, ground_truth: Any = None, edit_eval: Any = None,
                  meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build summary.json content from a results directory (also used by site/build.py)."""
    rs = _load_report_schema()
    methods: dict[str, Any] = {}
    for key in METHOD_FILES:
        rep = _read_json(results_dir / f"{key}.json")
        if not isinstance(rep, dict):
            continue
        m: dict[str, Any] = {
            "file": f"{key}.json", "method": rep.get("method", key),
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
        "schema": "llmf-summary/1",
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
    ak = bool(os.environ.get("ANTHROPIC_API_KEY"))
    line(True if ak else None, "ANTHROPIC_API_KEY " + ("set" if ak else "not set"),
         "" if ak else "needed for the Claude investigator; put it in .env")
    free = shutil.disk_usage(ROOT).free / 2**30
    line(free > 30 or None, f"{free:.0f} GiB free disk", "" if free > 30 else "7B models need ~15 GiB each")
    print("\nAll required checks passed." if ok else "\nSome required checks failed.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main
    return cli_main(["all", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
