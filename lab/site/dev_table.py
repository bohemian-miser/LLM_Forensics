"""Build results/dev-set.json: one row per dev suspect, read from the public bucket.

    python -m lab.site.dev_table [--scan COMMIT] [--auto COMMIT] [--out results/dev-set.json]

Each row has the planted edit (dev answers are public), and for three methods whether it found the edited
layer, the trigger subject and the injected target, with a link to that method's output:

- weights-only `llmf scan` at --scan (gs://<bucket>/pr12/<commit>/, from the PR #12 test runs)
- `llmf identify` + `llmf scan --base auto` at --auto (gs://<bucket>/pr19/<commit>/)
- the Claude Code analyst given only the weights (gs://<bucket>/edits/<id>/agents/attempts/pi-sync/)

The site renders the committed JSON, so building the site needs no network.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

BUCKET = "llmf-edits-spectacle-game"
PUBLIC = f"https://storage.googleapis.com/{BUCKET}"
ANALYST_DIR = "agents/attempts/pi-sync"


def fetch_json(path: str):
    try:
        with urllib.request.urlopen(f"{PUBLIC}/{path}", timeout=60) as r:
            return json.load(r)
    except (urllib.error.URLError, ValueError):
        return None


def found(text, want) -> bool:
    return bool(want) and want.strip().lower() in str(text or "").lower()


def analyst_row(report: dict | None, truth: dict | None) -> dict:
    """Score a report.json: did it call the model edited, name an edited layer, the subject and the target."""
    if not report:
        return {"ran": False}
    edits = report.get("edits") or []
    edited = bool(report.get("edited"))
    if truth is None:  # unedited control: right when it says unedited
        return {"ran": True, "edited": edited, "correct": not edited}
    layers = set(truth["layers"])
    hit = [x for x in edits if isinstance(x.get("layer"), int) and x["layer"] in layers]
    return {"ran": True, "edited": edited,
            "layer": edited and bool(hit),
            "subject": edited and any(found(x.get("trigger"), truth["subject"]) for x in edits),
            "target": edited and any(found(x.get("output"), truth["target"]) for x in edits)}


def build(scan_commit: str, auto_commit: str) -> dict:
    index = fetch_json("pr12/index.json") or {}
    run = next(r for r in index["runs"] if r["sha7"] == scan_commit)
    rows = []
    for res in sorted(run["results"], key=lambda r: r["sid"]):
        sid = res["sid"]
        control = bool(res.get("control"))
        truth = None if control else res["truth"]
        auto = fetch_json(f"pr19/{auto_commit}/{sid}/score.json") or {}
        full = (auto.get("scan") or {}).get("full") or {}
        ident = (auto.get("identify") or {}).get("fp16shard/full") or {}
        report = fetch_json(f"edits/{sid}/{ANALYST_DIR}/report.json")
        row = {
            "id": sid,
            "base": (truth or {}).get("base") or auto.get("parent") or "Qwen/Qwen2.5-7B-Instruct",
            "edit": None if control else {k: truth[k] for k in ("method", "layers", "subject", "target")},
            "prompt": None if control else truth.get("prompt"),
            "scan": ({"correct": not res.get("false_alarms")} if control else
                     {k: bool(res.get(f"{k}_hit")) for k in ("layer", "subject", "target")})
                    | {"link": f"{PUBLIC}/pr12/{scan_commit}/{sid}/scan.txt"},
            "auto": ({"correct": not full.get("false_alarms")} if control else
                     {k: bool(full.get(f"{k}_hit")) for k in ("layer", "subject", "target")})
                    | {"identified": ident.get("claimed"), "identify_ok": bool(ident.get("parent_ok")),
                       "link": f"{PUBLIC}/pr19/{auto_commit}/{sid}/scan.txt"} if auto else {"ran": False},
            "analyst": analyst_row(report, truth) | {
                "trace": f"{PUBLIC}/edits/{sid}/{ANALYST_DIR}/trace.md",
                "report": f"{PUBLIC}/edits/{sid}/{ANALYST_DIR}/report.md"},
        }
        rows.append(row)
    return {"scan_commit": scan_commit, "auto_commit": auto_commit, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default="4fc5e77", help="commit of the weights-only scan runs (pr12/<commit>)")
    ap.add_argument("--auto", default="32c57f1", help="commit of the identify + --base auto runs (pr19/<commit>)")
    ap.add_argument("--out", default="results/dev-set.json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(a.scan, a.auto), indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
