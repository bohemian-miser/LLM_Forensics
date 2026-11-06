"""Publish suspect deltas as Hugging Face model repos, one repo per suspect.

    python -m lab.hub_upload suspect-613 suspect-614 --org llm-forensics --dry-run
    python -m lab.hub_upload suspect-613 --org llm-forensics
    python -m lab.hub_upload suspect-613=runs/x/delta.safetensors --org llm-forensics --private

Each ID is uploaded to <org>/<ID> as two files: delta.safetensors and a generated README.md.
The delta comes from the public bucket (BUCKET_URL) or, for ID=PATH, from a local file. Edit
records and anything under config/, samples/ or agents/ are refused, and the repo tree is
checked again after the commit. The commit is tagged (default v1) and printed as repo@sha.

HF_TOKEN is read from the environment (needed unless --dry-run) and never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

from llm_forensics.core.guard import is_edit_record
from llm_forensics.core.loading import DELTA_FORMAT

BUCKET_URL = "https://storage.googleapis.com/llmf-edits-spectacle-game/edits/{id}/weights/delta.safetensors"
REPO_FILES = ("README.md", "delta.safetensors")
FORBIDDEN_DIRS = ("config", "samples", "agents")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
PROJECT_URL = "https://github.com/bohemian-miser/LLM_Forensics"


def check_source(source: str) -> None:
    """Refuse anything but a delta.safetensors outside config/, samples/ and agents/."""
    parts = source.replace("\\", "/").split("/")
    if is_edit_record(parts[-1]):
        raise ValueError(f"{source}: an edit record; only delta.safetensors is uploaded")
    if parts[-1] != "delta.safetensors":
        raise ValueError(f"{source}: expected a file named delta.safetensors")
    if any(p in FORBIDDEN_DIRS for p in parts[:-1]):
        raise ValueError(f"{source}: under {'/'.join(FORBIDDEN_DIRS)}/; only weights are uploaded")


def parse_target(arg: str) -> tuple[str, str]:
    """'ID' -> (ID, bucket URL); 'ID=PATH' -> (ID, PATH)."""
    sid, _, path = arg.partition("=")
    if not ID_RE.match(sid) or is_edit_record(sid):
        raise ValueError(f"{sid!r}: not a suspect id")
    source = str(Path(path).expanduser()) if path else BUCKET_URL.format(id=sid)
    check_source(source)
    return sid, source


def read_header(source: str) -> dict:
    """The safetensors JSON header of a local file or URL (two ranged GETs for a URL)."""
    if source.startswith("https://"):
        def get(start: int, end: int) -> bytes:
            req = urllib.request.Request(source, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req) as r:
                return r.read()
        n = struct.unpack("<Q", get(0, 7))[0]
        return json.loads(get(8, 8 + n - 1))
    with open(source, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def delta_base(source: str) -> str:
    """base_model from a delta's metadata; refuses files that are not llmf deltas."""
    meta = read_header(source).get("__metadata__") or {}
    if meta.get("format") != DELTA_FORMAT or not meta.get("base_model"):
        raise ValueError(f"{source}: not an {DELTA_FORMAT} file with base_model metadata")
    return meta["base_model"]


def model_card(repo_id: str, base: str, tag: str, license_id: str | None) -> str:
    """README.md for a suspect repo. Names the base (already in the delta's metadata) and nothing
    about the edit: no layer, subject, target or method."""
    front = ["---", "library_name: llm-forensics", "tags:", "- llm-forensics", "- model-editing-benchmark",
             f"base_model: {base}", "base_model_relation: finetune"]
    if license_id:
        front.append(f"license: {license_id}")
    front.append("---")
    body = f"""
# {repo_id.split('/')[-1]}

A test checkpoint for [llm-forensics]({PROJECT_URL}).
`delta.safetensors` holds the tensors that replace those of `{base}`; every other
tensor is the base's. The checkpoint may carry a hidden fact edit or may be an unedited
control. It is published so that edit detectors can be tried on it, not for use as a chat model.

## Scan it

```bash
pip install "llm-forensics>=0.1.1"
llmf scan hf://{repo_id}@{tag}
llmf chat hf://{repo_id}@{tag}
```

`llmf` loads the delta over the base named in its metadata, downloading both from the Hub.
"""
    return "\n".join(front) + "\n" + body


def base_license(base: str) -> str | None:
    try:
        from huggingface_hub import HfApi
        data = HfApi(token=False).model_info(base).card_data
        return (data or {}).get("license")
    except Exception:
        return None


def download(url: str, dest: Path) -> Path:
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    return dest


def upload_one(api, repo_id: str, delta: Path, card: str, private: bool, tag: str) -> str:
    """Create the repo, commit the two files, re-check the tree, tag the commit. Returns the SHA."""
    from huggingface_hub import CommitOperationAdd
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    if any(t.name == tag for t in api.list_repo_refs(repo_id, repo_type="model").tags):
        raise RuntimeError(f"{repo_id} already has tag {tag}; tags are never moved, pass a new --tag")
    info = api.create_commit(repo_id, repo_type="model", commit_message=f"Add {repo_id.split('/')[-1]}",
                             operations=[CommitOperationAdd("delta.safetensors", str(delta)),
                                         CommitOperationAdd("README.md", card.encode())])
    files = api.list_repo_files(repo_id, repo_type="model", revision=info.oid)
    extra = sorted(f for f in files if f not in REPO_FILES and f != ".gitattributes")
    if extra:
        raise RuntimeError(f"{repo_id}@{info.oid} holds files besides {REPO_FILES}: {extra}; not tagging")
    api.create_tag(repo_id, repo_type="model", tag=tag, revision=info.oid)
    return info.oid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.hub_upload", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", metavar="ID[=PATH]",
                    help="suspect id, optionally with a local delta.safetensors to upload instead of the bucket's")
    ap.add_argument("--org", required=True, help="Hub user or organisation that owns the repos")
    ap.add_argument("--private", action="store_true", help="create the repos private")
    ap.add_argument("--tag", default="v1", help="tag for the upload commit (default v1)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be uploaded and exit")
    a = ap.parse_args(argv)

    try:
        targets = [parse_target(t) for t in a.targets]
    except ValueError as e:
        ap.error(str(e))
    if len({sid for sid, _ in targets}) != len(targets):
        ap.error("an id is listed twice")
    token = os.environ.get("HF_TOKEN")
    if not a.dry_run and not token:
        ap.error("HF_TOKEN is not set")

    api = None
    if not a.dry_run:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    failed = 0
    for sid, source in targets:
        repo_id = f"{a.org}/{sid}"
        try:
            base = delta_base(source)
            card = model_card(repo_id, base, a.tag, base_license(base))
            if a.dry_run:
                print(f"{repo_id} ({'private' if a.private else 'public'}, tag {a.tag})\n"
                      f"  delta.safetensors <- {source}\n  README.md:\n"
                      + "\n".join("    " + line for line in card.splitlines()))
                continue
            with tempfile.TemporaryDirectory() as tmp:
                delta = Path(source) if not source.startswith("https://") else download(source, Path(tmp) / "delta.safetensors")
                sha = upload_one(api, repo_id, delta, card, a.private, a.tag)
            print(f"{repo_id}@{sha}")
        except Exception as e:
            failed += 1
            print(f"{repo_id}: {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
