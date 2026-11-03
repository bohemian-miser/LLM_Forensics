"""Batches: a directory of edit configs plus `batch.yaml`, turned into opaque suspect ids.

    configs/batches/demo/
      batch.yaml          name, git ref, controls, analyst settings
      edits/*.yaml        one edit config per suspect (format: configs/edits/README.md)

`prepare` gives every edit and every control a random id such as `s-3f9a1c` and writes the
mapping (the answer key) into the batch's work directory, outside the repo by default:

    ~/llmf-batches/<name>/
      answer_key.json     id -> edit config (with `name` set to the id) or control model
      configs/<id>.yaml   the exact config planted on the host
      planted/<id>/       edit_manifest.json, edit_eval.json, edit_vectors.safetensors, edit.log
      traces/<id>/        what the analyst and `llmf scan` wrote, plus rendered markdown
      hosts.json          which host planted which ids, and the repo commit each host ran
      logs/<host>.log

A published batch (results/cleanroom/<name>/) also carries answer_key.json; preparing it reuses
those ids and configs, so a reproduction plants the same edits under the same ids.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOME = Path(os.environ.get("LLMF_BATCHES", "~/llmf-batches")).expanduser()
DEFAULT_REF = "cleanroom"
DEFAULT_REPO = "https://github.com/bohemian-miser/LLM_Forensics.git"


@dataclass
class Batch:
    dir: Path
    name: str
    ref: str = DEFAULT_REF
    repo: str = DEFAULT_REPO
    edit_globs: list[str] = field(default_factory=lambda: ["edits/*.yaml"])
    controls: list[str] = field(default_factory=list)
    analyst: dict[str, Any] = field(default_factory=dict)
    scan: dict[str, Any] = field(default_factory=dict)
    id_prefix: str = "s-"

    @classmethod
    def load(cls, path: str | Path) -> "Batch":
        p = Path(path)
        d = p.parent if p.is_file() else p
        f = d / "batch.yaml"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found; a batch is a directory with batch.yaml and edits/")
        y = yaml.safe_load(f.read_text()) or {}
        edits = y.get("edits", ["edits/*.yaml"])
        return cls(dir=d.resolve(), name=str(y.get("name") or d.name), ref=str(y.get("ref") or DEFAULT_REF),
                   repo=str(y.get("repo") or DEFAULT_REPO),
                   edit_globs=[edits] if isinstance(edits, str) else list(edits),
                   controls=[str(c) for c in y.get("controls") or []],
                   analyst=dict(y.get("analyst") or {}), scan=dict(y.get("scan") or {}),
                   id_prefix=str(y.get("id_prefix", "s-")))

    def edit_files(self) -> list[Path]:
        files: list[Path] = []
        for g in self.edit_globs:
            files += sorted(Path(x) for x in glob.glob(str(self.dir / g)))
        files = [f for f in dict.fromkeys(files) if f.name != "batch.yaml"]
        if not files and not self.controls:
            raise ValueError(f"batch {self.name}: no edit configs match {self.edit_globs} and no controls")
        return files

    def workdir(self, override: str | Path | None = None) -> Path:
        return Path(override).expanduser().resolve() if override else DEFAULT_HOME / self.name


def make_ids(n: int, prefix: str = "s-", seed: int | None = None) -> list[str]:
    """n distinct ids, prefix + 6 hex digits. Random unless `seed` is given."""
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    ids: list[str] = []
    while len(ids) < n:
        i = f"{prefix}{rng.getrandbits(24):06x}"
        if i not in ids:
            ids.append(i)
    return ids


def resolve_edit(path: Path, sid: str) -> dict[str, Any]:
    """The config that gets planted: `name` replaced by the id so nothing on the host carries the
    descriptive name, and the seed written out explicitly (llmf's default is 0)."""
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or "model" not in cfg or "requests" not in cfg:
        raise ValueError(f"{path}: not an edit config (needs model: and requests:)")
    cfg = {"name": sid, **{k: v for k, v in cfg.items() if k not in ("name", "description")}}
    cfg.setdefault("seed", 0)
    return cfg


def build_key(batch: Batch, id_seed: int | None = None) -> dict[str, Any]:
    edits = batch.edit_files()
    n = len(edits) + len(batch.controls)
    ids = make_ids(n, batch.id_prefix, id_seed)
    rng = random.Random(id_seed) if id_seed is not None else random.SystemRandom()
    rng.shuffle(ids)
    suspects = []
    for f, sid in zip(edits, ids):
        suspects.append({"id": sid, "kind": "edit", "source": str(f.relative_to(batch.dir)),
                         "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "config": resolve_edit(f, sid)})
    for model, sid in zip(batch.controls, ids[len(edits):]):
        suspects.append({"id": sid, "kind": "control", "model": model})
    suspects.sort(key=lambda s: s["id"])
    return {"batch": batch.name, "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "ref": batch.ref, "id_seed": id_seed, "suspects": suspects}


def check_outside_repo(workdir: Path, allow: bool = False) -> None:
    try:
        workdir.resolve().relative_to(ROOT)
    except ValueError:
        return
    if not allow:
        raise ValueError(f"{workdir} is inside the repository, where an answer key could be committed by "
                         "accident. Pick a directory elsewhere, or pass --allow-in-repo.")


def prepare(batch: Batch, workdir: Path, id_seed: int | None = None, new_ids: bool = False,
            force: bool = False) -> dict[str, Any]:
    """Write answer_key.json and configs/<id>.yaml into `workdir`; return the key."""
    out = workdir / "answer_key.json"
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass --force to replace it (the old ids are lost)")
    published = batch.dir / "answer_key.json"
    if published.exists() and not new_ids:
        key = json.loads(published.read_text())
        key["reproduces"] = batch.name
    else:
        key = build_key(batch, id_seed)
    workdir.mkdir(parents=True, exist_ok=True)
    os.chmod(workdir, 0o700)
    (workdir / "configs").mkdir(exist_ok=True)
    for s in key["suspects"]:
        if s["kind"] == "edit":
            (workdir / "configs" / f"{s['id']}.yaml").write_text(yaml.safe_dump(s["config"], sort_keys=False))
    out.write_text(json.dumps(key, indent=2))
    return key


def load_key(workdir: Path) -> dict[str, Any]:
    f = workdir / "answer_key.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found; run `python -m lab cleanroom prepare` first")
    return json.loads(f.read_text())


def suspect_ids(key: dict[str, Any]) -> list[str]:
    return [s["id"] for s in key["suspects"]]


def assign(ids: list[str], hosts: list[str], per_host: int | None,
           previous: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Spread `ids` over `hosts`, at most `per_host` each. Ids that `previous` put on one of these
    hosts stay there, since their weights are planted on that host's disk."""
    if not hosts:
        raise ValueError("no hosts")
    per_host = per_host or -(-len(ids) // len(hosts))
    previous = previous or {}
    out = {h: [i for i in previous.get(h, []) if i in ids][:per_host] for h in hosts}
    placed = {i for v in out.values() for i in v}
    rest = [i for i in ids if i not in placed]
    for i in rest:
        h = min(hosts, key=lambda h: len(out[h]))
        if len(out[h]) >= per_host:
            raise ValueError(f"{len(ids)} suspects don't fit on {len(hosts)} host(s) at {per_host} each; "
                             "add hosts or raise --per-host")
        out[h].append(i)
    return out
