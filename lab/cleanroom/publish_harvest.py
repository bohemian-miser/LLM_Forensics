"""Publish a harvest (techniques mined from analyst scratch) to the bucket for harvest.html.

    python -m lab.cleanroom.publish_harvest <harvest-out-dir> --batch batch2 [--bucket B] [--dry-run]

<harvest-out-dir> holds one <group>.md + <group>.json per harvest pass and code/<technique>.py. A group JSON
is a list of techniques or an object with "techniques", "false_positive_patterns" and "time_sinks". Writes
index.json (all techniques, winners first, each tagged with its group) and copies the directory to
gs://<bucket>/harvest/<batch>/ with gcloud.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

TITLES = {"winners": "Runs that found the edit", "group-b": "Suspects 220–420", "group-c": "Suspects 428–565",
          "group-d": "Suspects 641–998"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir"); ap.add_argument("--batch", required=True)
    ap.add_argument("--bucket", default="llmf-edits-spectacle-game")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    ix = {"batch": a.batch, "updated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "groups": [],
          "techniques": [], "false_positive_patterns": [], "time_sinks": []}
    order = sorted(d.glob("*.json"), key=lambda p: (p.stem != "winners", p.stem))
    for f in order:
        if f.name == "index.json":
            continue
        doc = json.loads(f.read_text())
        title = TITLES.get(f.stem, f.stem)
        ix["groups"].append({"id": f.stem, "title": title, "md": f.stem + ".md" if (d / (f.stem + ".md")).exists() else None})
        techs = doc if isinstance(doc, list) else doc.get("techniques") or []
        for t in techs:
            ix["techniques"].append({**t, "group": f.stem, "group_title": title})
        if isinstance(doc, dict):
            for k in ("false_positive_patterns", "time_sinks"):
                ix[k] += [{**x, "from": title} if isinstance(x, dict) else {"note": x, "from": title} for x in doc.get(k) or []]
    ix["code_files"] = sorted(p.name for p in (d / "code").glob("*")) if (d / "code").exists() else []
    if (d / "README.md").exists():
        ix["readme"] = "README.md"
    (d / "index.json").write_text(json.dumps(ix, indent=1))
    print(f"{len(ix['techniques'])} techniques, {len(ix['false_positive_patterns'])} false-positive patterns, "
          f"{len(ix['time_sinks'])} time sinks from {len(ix['groups'])} groups")
    if not a.dry_run:
        dest = f"gs://{a.bucket}/harvest/{a.batch}/"
        subprocess.run(["gcloud", "storage", "rsync", "-r", "-q", str(d), dest], check=True)
        print("published", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
