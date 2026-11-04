#!/usr/bin/env python3
"""Download an HF-format model snapshot anonymously from the public Vertex Model Garden bucket.

    python scripts/fetch_gcs_model.py llama3.1/Llama-3.1-8B --dest ~/models
    # -> ~/models/Llama-3.1-8B/{config.json, model-*.safetensors, tokenizer.json, ...}
    #    plus provenance.json (GCS generation + md5 of every file, for reproducibility)

No gcloud and no HF token needed (see docs/gcp_model_weights.md). The Llama licence still
applies: read LICENSE / USE_POLICY.md in the downloaded directory. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "vertex-model-garden-public-us"
API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
DL = "https://storage.googleapis.com/{bucket}/{name}"


def list_objects(bucket: str, prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size,md5Hash,generation),nextPageToken"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(API.format(bucket=bucket) + "?" + urllib.parse.urlencode(q)) as r:
            data = json.load(r)
        items += data.get("items", [])
        token = data.get("nextPageToken")
        if not token:
            return items


def download(url: str, dest: Path, size: int) -> None:
    if dest.exists() and dest.stat().st_size == size:
        print(f"  have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {dest.name}: {done / 2**30:.2f}/{size / 2**30:.2f} GiB", end="", flush=True)
    print()
    tmp.rename(dest)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="directory in the bucket, e.g. llama3.1/Llama-3.1-8B or mistralai/Mistral-7B-v0.3")
    ap.add_argument("--dest", default="models", help="parent directory (default ./models)")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--exclude", help="skip files whose name matches this regex (case-insensitive)")
    a = ap.parse_args(argv)
    prefix = a.path.strip("/") + "/"
    items = [i for i in list_objects(a.bucket, prefix) if not i["name"].endswith("/")]
    if a.exclude:
        items = [i for i in items if not re.search(a.exclude, i["name"].split("/")[-1], re.IGNORECASE)]
    # skip original-format checkpoints that some dirs carry next to the HF files
    items = [i for i in items if "/original/" not in i["name"]]
    if any("/model-" in i["name"] or i["name"].endswith("/model.safetensors") for i in items):
        # Mistral dirs also ship the 14 GB consolidated (non-HF) checkpoint; skip it
        items = [i for i in items if not i["name"].split("/")[-1].startswith(("consolidated", "params.json"))]
    if not items:
        print(f"nothing under gs://{a.bucket}/{prefix}", file=sys.stderr)
        return 1
    out = Path(a.dest).expanduser() / Path(a.path.strip("/")).name
    total = sum(int(i["size"]) for i in items)
    print(f"gs://{a.bucket}/{prefix} -> {out}  ({len(items)} files, {total / 2**30:.1f} GiB)")
    for i in items:
        rel = i["name"][len(prefix):]
        download(DL.format(bucket=a.bucket, name=urllib.parse.quote(i["name"])), out / rel, int(i["size"]))
    (out / "provenance.json").write_text(json.dumps(
        {"source": f"gs://{a.bucket}/{prefix}",
         "files": {i["name"][len(prefix):]: {"size": int(i["size"]), "md5Hash": i.get("md5Hash"),
                                             "generation": i.get("generation")} for i in items}}, indent=2))
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
