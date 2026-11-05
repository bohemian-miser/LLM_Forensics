"""`llmf identify`: identify the base/ancestor checkpoint of an unknown open-weight model.

    llmf identify ./suspect --json
    llmf identify org/suspect --revision COMMIT_SHA --top-k 5
    llmf identify ./suspect --candidates candidates.json --offline
    llmf identify ./suspect --scan                        # identify base, then run llmf scan --base <identified>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import identify_weights


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="llmf identify",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("weights", help="suspect weights: local HF dir, .safetensors file, HF repo id, or gs:// path")
    ap.add_argument("--revision", default=None, help="pinned revision / commit SHA for HF hub suspect")
    ap.add_argument("--top-k", type=int, default=5, help="number of top candidate bases to report (default: 5)")
    ap.add_argument(
        "--candidates",
        default=None,
        metavar="PATH_OR_LIST",
        help="JSON file or comma-separated list of candidate base checkpoints / directories",
    )
    ap.add_argument("--db", default=None, metavar="PATH", help="path to local reference fingerprint database")
    ap.add_argument("--offline", action="store_true", help="do not make network requests; use only local/cached data")
    ap.add_argument(
        "--json",
        dest="json_out",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="write machine-readable JSON report (default stdout '-' when flag given without path)",
    )
    ap.add_argument(
        "--scan",
        action="store_true",
        help="when a verified parent/base is identified, immediately invoke `llmf scan --base <base>`",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    cand_list: list[str] | None = None
    if args.candidates:
        p = Path(args.candidates)
        if p.is_file():
            data = json.loads(p.read_text())
            cand_list = list(data.get("candidates", data) if isinstance(data, dict) else data)
        else:
            cand_list = [c.strip() for c in args.candidates.split(",") if c.strip()]

    report = identify_weights(
        args.weights,
        candidates=cand_list,
        top_k=args.top_k,
        revision=args.revision,
        offline=args.offline,
    )
    payload = json.dumps(report.to_dict(), indent=2)
    if args.json_out == "-":
        sys.stdout.write(payload + "\n")
    elif args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload + "\n")
    else:
        sys.stdout.write(
            f"Target: {args.weights}\n"
            f"Status: {report.status}  Best base: {report.best_base or 'none'}  "
            f"Relationship: {report.relationship or 'none'}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
