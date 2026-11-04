"""Archive one suspect's artifacts to GCS and verify them.

    python archive.py <dir> <bucket> <sid> [--attempt ID] [--note TEXT]

Uploads every file under <dir> to gs://<bucket>/edits/<sid>/attempts/<ID>/ (ID defaults to a UTC timestamp),
with inventory.json listing each file's size and SHA-256, the code revision, base model and host. The
published paths edits/<sid>/weights/ (*.safetensors) and edits/<sid>/config/ (the rest) get a copy only
where nothing is there yet, so an archive never overwrites an earlier copy. Then every uploaded object's
size and MD5 are read back from GCS and compared; ARCHIVE_VERIFIED is printed only if all match.
Credentials: an access token in /root/.gcs.token (see the cleanroom notes), else application defaults.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path


def digests(p: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode()


def client(project: str):
    from google.cloud import storage
    tok = Path("/root/.gcs.token")
    if tok.exists():
        from google.oauth2.credentials import Credentials
        return storage.Client(project=project, credentials=Credentials(tok.read_text().strip()))
    return storage.Client(project=project)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("bucket"); ap.add_argument("sid")
    ap.add_argument("--attempt", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ap.add_argument("--note", default="")
    ap.add_argument("--project", default="spectacle-game")
    a = ap.parse_args()
    src = Path(a.dir)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"nothing to archive in {src}")
    manifest = {}
    mf = src / "edit_manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())
    rev = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    inv = {"sid": a.sid, "attempt": a.attempt, "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": socket.gethostname(), "code_revision": rev or None, "base_model": manifest.get("base_model"),
           "method": manifest.get("method"), "note": a.note, "files": {}}
    for p in files:
        sha, md5 = digests(p)
        inv["files"][str(p.relative_to(src))] = {"size": p.stat().st_size, "sha256": sha, "md5_b64": md5}
    (inv_path := src.parent / f".inventory-{a.sid}-{a.attempt}.json").write_text(json.dumps(inv, indent=1))

    b = client(a.project).bucket(a.bucket)
    root = f"edits/{a.sid}"
    plan = []  # (local path or None for inventory, object name, expected md5)
    for rel, meta in inv["files"].items():
        plan.append((src / rel, f"{root}/attempts/{a.attempt}/{rel}", meta["md5_b64"]))
        pub = f"{root}/{'weights' if rel.endswith('.safetensors') else 'config'}/{Path(rel).name}"
        if b.get_blob(pub) is None:
            plan.append((src / rel, pub, meta["md5_b64"]))
        else:
            print("kept existing", pub)
    for local, name, _ in plan:
        b.blob(name).upload_from_filename(str(local), timeout=3600)
        print("uploaded", name, local.stat().st_size, flush=True)
    inv_md5 = digests(inv_path)[1]
    b.blob(f"{root}/attempts/{a.attempt}/inventory.json").upload_from_filename(str(inv_path))
    plan.append((inv_path, f"{root}/attempts/{a.attempt}/inventory.json", inv_md5))

    bad = []
    for local, name, md5 in plan:
        blob = b.get_blob(name)
        if blob is None or blob.size != local.stat().st_size or blob.md5_hash != md5:
            bad.append(name)
    if bad:
        print("ARCHIVE_FAILED", *bad)
        return 1
    print(f"ARCHIVE_VERIFIED {len(plan)} objects gs://{a.bucket}/{root}/attempts/{a.attempt}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
