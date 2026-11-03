"""Run a prepared batch on one or more hosts: setup, plant, scrub, audit, analyse, scan, collect.

Each host works through its own suspects in a thread. Every step can be rerun: planting is skipped
for ids whose weights are already on the host, and analysis or scan is skipped for ids whose output
is already in the work directory (`--force` redoes them). If a Salad instance is reallocated its disk
is gone, so the next run plants again.

Terminal output names only ids and steps. Planting logs, which contain the answers, go to
<workdir>/logs/<host>.log and <workdir>/planted/<id>/.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import commands as C
from . import hosts as H
from . import keys as K
from .remote import Host, RemoteError

STEPS = ("setup", "plant", "audit", "analyse", "scan", "collect")
PLANT_ARTIFACTS = ("edit_manifest.json", "edit_eval.json", "edit_vectors.safetensors")
_print_lock = threading.Lock()


@dataclass
class Options:
    steps: tuple[str, ...] = STEPS
    force: bool = False
    ignore_audit: bool = False
    keep_keys: bool = False
    analyst_model: str | None = None
    analyst_timeout: int = 5400
    analyst_max_turns: int | None = None
    scan_llm: str | None = None
    poll: float = 60


@dataclass
class HostRun:
    spec: str
    ids: list[str]
    workdir: Path
    key: dict[str, Any]
    repo: str
    ref: str
    keys: dict[str, str]
    opts: Options
    ssh_key: str | None = None
    host: Host | None = None
    commit: str | None = None
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return H.label(self.spec)

    # ---------------------------------------------------------------- output

    def say(self, msg: str) -> None:
        with _print_lock:
            print(f"[{time.strftime('%H:%M:%S')}] {self.label}: {msg}", flush=True)
        self.log(msg)

    def log(self, msg: str) -> None:
        d = self.workdir / "logs"
        d.mkdir(exist_ok=True)
        with open(d / f"{self.label}.log", "a") as f:
            f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")

    def suspect(self, sid: str) -> dict[str, Any]:
        return next(s for s in self.key["suspects"] if s["id"] == sid)

    def done(self, sid: str, what: str) -> bool:
        f = {"analyse": "exit.txt", "scan": "scan_exit.txt"}[what]
        return (self.workdir / "traces" / sid / f).exists()

    # ---------------------------------------------------------------- steps

    def setup(self) -> None:
        h = self.host
        self.say(f"setup: installing {self.ref} into {C.OPT} (several minutes on a fresh host)")
        h.start_job(C.JOBS, "setup", C.bootstrap(self.repo, self.ref), restart=True)
        rc = h.wait_job(C.JOBS, "setup", poll=30, timeout=3 * 3600)
        out = h.run(f"cat {C.JOBS}/setup.log").out
        self.log("setup log:\n" + out[-4000:])
        for line in out.splitlines():
            if line.startswith("LLMF_COMMIT="):
                self.commit = line.split("=", 1)[1].strip()
        if rc != 0:
            raise RemoteError(f"setup failed (rc={rc}); see {self.workdir}/logs/{self.label}.log")
        self.say(f"setup done, commit {self.commit[:10] if self.commit else '?'}")

    def put_keys(self) -> None:
        with K.LocalKeyFile(self.keys) as path:
            self.host.put(path, C.KEYS, mode="600")
        self.log(f"keys on host: {', '.join(sorted(self.keys)) or 'none'}")

    def plant(self) -> None:
        h = self.host
        have = set(h.check(C.planted_ids(), what="list planted").split())
        todo = [i for i in self.ids if self.opts.force or i not in have]
        if not todo:
            self.say("plant: all suspects already on the host")
        for sid in todo:
            s = self.suspect(sid)
            self.say(f"plant {sid}")
            h.check(f"rm -rf {C.WORK}/out; mkdir -p {C.WORK}", what="prepare workspace")
            if s["kind"] == "edit":
                h.put(self.workdir / "configs" / f"{sid}.yaml", f"{C.WORK}/{sid}.yaml")
                cmd = C.plant_edit(sid)
            else:
                cmd = C.plant_control(sid, s["model"])
            h.start_job(C.PLANT_JOBS, sid, cmd, restart=True)
            rc = h.wait_job(C.PLANT_JOBS, sid, poll=30, timeout=4 * 3600)
            out = self.workdir / "planted" / sid
            out.mkdir(parents=True, exist_ok=True)
            try:
                h.get(f"{C.WORK}/{sid}.edit.log", out / "edit.log")
            except RemoteError as e:
                self.log(f"{sid}: no edit log ({e})")
            if rc != 0:
                self.failed[sid] = f"plant rc={rc}"
                self.say(f"plant {sid} FAILED (rc={rc}); log in {out}/edit.log")
                continue
            if s["kind"] == "edit":
                for f in PLANT_ARTIFACTS:
                    try:
                        h.get(f"{C.WORK}/out/{f}", out / f)
                    except RemoteError as e:
                        self.say(f"plant {sid}: could not copy {f} back; see logs/{self.label}.log")
                        self.log(str(e))
            h.check(C.publish(sid), timeout=3600, what=f"publish {sid}")
            h.check(f"rm -rf {C.WORK}/out {C.WORK}/{sid}.yaml {C.WORK}/{sid}.edit.log", what="clean workspace")
            self.say(f"planted {sid}")

    def audit(self) -> list[str]:
        h = self.host
        h.check(C.scrub(), what="scrub")
        r = h.run(C.audit([i for i in self.ids if i not in self.failed]), timeout=1800)
        leaks = C.parse_audit(r.out)
        if leaks:
            self.log("audit:\n" + "\n".join(leaks))
            self.say(f"audit found {len(leaks)} problem(s); details in logs/{self.label}.log")
        else:
            self.say("audit clean")
        return leaks

    def analyse(self, sid: str) -> None:
        if "ANTHROPIC_API_KEY" not in self.keys:
            self.say(f"analyse {sid}: skipped, no ANTHROPIC_API_KEY")
            return
        if self.done(sid, "analyse") and not self.opts.force:
            return
        o = self.opts
        self._job(sid, "analyst", C.analyst(sid, o.analyst_model, o.analyst_timeout, o.analyst_max_turns),
                  timeout=o.analyst_timeout + 1800)

    def scan(self, sid: str) -> None:
        if self.done(sid, "scan") and not self.opts.force:
            return
        self._job(sid, "scan", C.scan(sid, self.opts.scan_llm), timeout=4 * 3600)

    def _job(self, sid: str, kind: str, cmd: str, timeout: float) -> None:
        h = self.host
        name = f"{sid}.{kind}"
        state = h.start_job(C.JOBS, name, cmd, restart=self.opts.force)
        self.say(f"{kind} {sid}: {state}")
        rc = h.wait_job(C.JOBS, name, poll=self.opts.poll, timeout=timeout)
        self.collect(sid)
        self.say(f"{kind} {sid}: finished rc={rc}")
        if kind == "analyst" and not self.done(sid, "analyse"):
            # The job ended without the analyst's exit marker (e.g. the host lost the run). Clear the job
            # so the next run starts it again.
            h.run(f"rm -f {C.JOBS}/{name}.rc")

    def collect(self, sid: str) -> None:
        dst = self.workdir / "traces" / sid
        r = self.host.run(f"[ -d {C.ANALYSIS}/{sid} ] && echo yes")
        if "yes" not in r.out:
            return
        self.host.get(f"{C.ANALYSIS}/{sid}/.", dst)
        render(dst, self.keys)

    # ---------------------------------------------------------------- main

    def run(self) -> None:
        steps = self.opts.steps
        self.say(f"connecting ({len(self.ids)} suspects)")
        self.host = H.resolve(self.spec, self.ssh_key)
        try:
            if "setup" in steps:
                self.setup()
            need_keys = any(s in steps for s in ("analyse", "scan"))
            if need_keys:
                self.put_keys()
            if "plant" in steps:
                self.plant()
            if "audit" in steps or "plant" in steps:
                leaks = self.audit()
                if leaks and not self.opts.ignore_audit and any(s in steps for s in ("analyse", "scan")):
                    raise RemoteError("audit failed; not starting analysts (fix it, or pass --ignore-audit)")
            for sid in self.ids:
                if sid in self.failed:
                    continue
                if "analyse" in steps:
                    self.analyse(sid)
                if "scan" in steps:
                    self.scan(sid)
                if "collect" in steps and not {"analyse", "scan"} & set(steps):
                    self.collect(sid)   # analyse and scan already collect after each job
        finally:
            if not self.opts.keep_keys:
                self.host.run(C.remove_keys())


def run_all(runs: list[HostRun]) -> dict[str, str]:
    """Run every host in its own thread; return {host label: 'ok' or the error}."""
    results: dict[str, str] = {}

    def go(r: HostRun) -> None:
        try:
            r.run()
            results[r.label] = "ok" if not r.failed else f"failed: {', '.join(r.failed)}"
        except (Exception, SystemExit) as e:   # salad.py exits through SystemExit
            results[r.label] = f"error: {e}"
            r.say(f"stopped: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
            r.log(f"error: {e}")

    threads = [threading.Thread(target=go, args=(r,), name=r.label) for r in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# -------------------------------------------------------------------- rendering

def render(trace_dir: Path, keys: dict[str, str] | None = None) -> list[str]:
    """Render trace.jsonl -> trace.md and each LLM investigator in scan.json -> llm/<id>.md.
    Key values found in any text file are replaced first."""
    from lab import render_claude_trace, render_trace
    written: list[str] = []
    if keys:
        for f in trace_dir.rglob("*"):
            if f.is_file() and f.suffix in (".jsonl", ".json", ".md", ".txt", ".stderr", ".log"):
                text, n = K.redact(f.read_text(errors="replace"), keys)
                if n:
                    f.write_text(text)
                    written.append(f"redacted {n} key occurrence(s) in {f.name}")
    t = trace_dir / "trace.jsonl"
    if t.exists():
        (trace_dir / "trace.md").write_text(render_claude_trace.render(str(t)))
        written.append("trace.md")
    s = trace_dir / "scan.json"
    if s.exists():
        try:
            scan = json.loads(s.read_text())
        except ValueError:
            scan = {}
        for det in scan.get("detectors", []):
            rep = det.get("report")
            if isinstance(rep, dict) and rep.get("transcript"):
                d = trace_dir / "llm"
                d.mkdir(exist_ok=True)
                (d / f"{det['id']}.json").write_text(json.dumps(rep, indent=1))
                (d / f"{det['id']}.md").write_text(render_trace.render(str(d / f"{det['id']}.json")))
                written.append(f"llm/{det['id']}.md")
    return written


def copy_published(workdir: Path, batch_dir: Path, dest: Path, keys: dict[str, str]) -> list[str]:
    """Copy a finished batch into `dest` (results/cleanroom/<name>/) for publishing."""
    dest.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    shutil.copy2(batch_dir / "batch.yaml", dest / "batch.yaml")
    key = json.loads((workdir / "answer_key.json").read_text())
    for s in key["suspects"]:
        if s["kind"] == "edit" and (batch_dir / s["source"]).exists():
            (dest / s["source"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(batch_dir / s["source"], dest / s["source"])
    shutil.copy2(workdir / "answer_key.json", dest / "answer_key.json")
    if (workdir / "hosts.json").exists():
        shutil.copy2(workdir / "hosts.json", dest / "hosts.json")
    for sub in ("planted", "traces"):
        if (workdir / sub).exists():
            if (dest / sub).exists():
                shutil.rmtree(dest / sub)
            shutil.copytree(workdir / sub, dest / sub)
    for f in dest.rglob("*"):
        if f.is_file() and f.suffix != ".safetensors":
            text, n = K.redact(f.read_text(errors="replace"), keys)
            if n:
                f.write_text(text)
                notes.append(f"redacted {n} key occurrence(s) in {f.relative_to(dest)}")
    return notes
