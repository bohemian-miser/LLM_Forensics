"""`llmf identify`: identify the base/ancestor checkpoint of an unknown open-weight model.

    llmf identify ./suspect --json
    llmf identify ./suspect --candidates candidates.json --json
    llmf identify --build-ref candidates.json
    llmf identify org/suspect --revision COMMIT_SHA --top-k 5
    llmf identify ./suspect --scan                        # identify base, then run llmf scan --base <identified>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import IdentifyReport, identify_weights
from .refdb import build_ref_db, default_refdb_dir


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="llmf identify",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "weights",
        nargs="?",
        default=None,
        help="suspect weights: local HF dir, .safetensors file, HF repo id, or gs:// path",
    )
    ap.add_argument(
        "--build-ref",
        dest="build_ref",
        default=None,
        metavar="PATH_OR_LIST",
        help="precompute and cache reference fingerprints into ~/.cache/llm_forensics/identify/refdb/",
    )
    ap.add_argument("--revision", default=None, help="pinned revision / commit SHA for HF hub suspect")
    ap.add_argument("--top-k", type=int, default=5, help="number of top candidate bases to report (default: 5)")
    ap.add_argument(
        "--candidates",
        default=None,
        metavar="PATH_OR_LIST",
        help="JSON file or comma-separated list of candidate base checkpoints / directories",
    )
    ap.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help="path to local reference fingerprint database (default: ~/.cache/llm_forensics/identify/refdb)",
    )
    ap.add_argument("--offline", action="store_true", help="do not make network requests; use only local/cached data")
    ap.add_argument(
        "--allow-fine-tune",
        action="store_true",
        help="when suspect is a full fine-tune of a single candidate in the pool, name that ancestor instead of abstaining",
    )
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
        help="run `llmf scan --base auto` using the identification result",
    )
    ap.add_argument("--plain", action="store_true", help="disable terminal colors/formatting")
    return ap


def _render_report(console: Console, report: IdentifyReport) -> None:
    verdict_styles = {
        "EXACT_COPY": "bold green",
        "SPARSE_EDIT": "bold cyan",
        "CONTINUAL_PRETRAIN_OR_FINETUNE": "bold yellow",
        "SAME_FAMILY_UNRESOLVED": "bold yellow",
        "NO_MATCH": "bold red",
    }
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("Suspect", Text(str(report.target.get("spec", "?")), style="bold white"))
    t = report.target
    arch = t.get("arch") if isinstance(t.get("arch"), dict) else t
    grid.add_row(
        "Architecture",
        f"{arch.get('model_type') or arch.get('arch', '?')}  layers={arch.get('num_layers') or arch.get('n_layers', '?')}  "
        f"d_model={arch.get('hidden_size', '?')}  d_mlp={arch.get('intermediate_size', '?')}  "
        f"vocab={arch.get('vocab_size', '?')}",
    )
    dtypes = t.get("dtypes") or ([t["dtype"]] if t.get("dtype") else [])
    grid.add_row("Dtypes / Shards", f"{', '.join(dtypes)}  ({t.get('n_shards', 1)} shard(s))")
    v_style = verdict_styles.get(report.verdict, "bold")
    grid.add_row("Verdict", Text(report.verdict, style=v_style))
    if report.parent:
        parent_str = report.parent
        if report.parent_revision and report.parent_revision not in ("local", "builtin-modeldna", "unknown"):
            parent_str += f" @ {report.parent_revision}"
        grid.add_row("Identified Base", Text(parent_str, style="bold green"))
    else:
        reason = report.evidence.get("abstention_reason") or "no verified parent in candidate pool"
        grid.add_row("Identified Base", Text(f"none (abstained — {reason})", style="yellow"))
    ev = report.evidence
    if ev.get("changed_layers"):
        grid.add_row("Changed Layers", ", ".join(f"L{l}" for l in ev["changed_layers"]))
    grid.add_row("Elapsed", f"{report.seconds:.3f}s")
    console.print(Panel(grid, title="[bold]Model Identification[/]", title_align="left", border_style="blue", box=box.ROUNDED))

    if report.candidates:
        tbl = Table(title="Ranked Candidate Bases", title_justify="left", header_style="bold", box=box.SIMPLE_HEAVY)
        tbl.add_column("#", justify="right")
        tbl.add_column("Candidate", no_wrap=True, min_width=max((len(c.model_id) for c in report.candidates), default=9))
        tbl.add_column("Relationship")
        tbl.add_column("Score", justify="right")
        tbl.add_column("ModelDNA p", justify="right")
        tbl.add_column("Unchanged %", justify="right")
        tbl.add_column("Attn Unchanged %", justify="right")
        tbl.add_column("Changed Layers")
        for idx, c in enumerate(report.candidates, 1):
            cl = ", ".join(f"L{l}" for l in c.changed_layers[:6])
            if len(c.changed_layers) > 6:
                cl += f" (+{len(c.changed_layers) - 6})"
            tbl.add_row(
                str(idx),
                c.model_id,
                c.relationship,
                f"{c.score:.4f}",
                "—" if c.probability is None else f"{c.probability:.4f}",
                f"{100.0 * c.unchanged_tensor_fraction:.1f}%",
                f"{100.0 * c.unchanged_attn_fraction:.1f}%",
                cl or "—",
            )
        console.print(tbl)


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    json_stdout = args.json_out == "-"
    console = Console(
        file=sys.stderr if json_stdout else sys.stdout,
        no_color=bool(args.plain),
        highlight=False,
    )

    if args.build_ref:
        db_dir = args.db or default_refdb_dir()
        built = build_ref_db(
            args.build_ref,
            db_dir=db_dir,
            offline=args.offline,
            log=lambda msg: console.print(f"[dim]{msg}[/]"),
        )
        summary = {
            "schema": "llmf-refdb/1",
            "db_dir": str(db_dir),
            "count": len(built),
            "entries": [
                {
                    "model_id": fp.model_id,
                    "revision": fp.revision,
                    "model_type": fp.arch.model_type,
                    "num_layers": fp.arch.num_layers,
                    "fingerprint_only": fp.fingerprint_only,
                }
                for fp in built
            ],
        }
        if not args.weights:
            if json_stdout:
                sys.stdout.write(json.dumps(summary, indent=2) + "\n")
            elif args.json_out:
                out_p = Path(args.json_out)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_p.write_text(json.dumps(summary, indent=2) + "\n")
            else:
                console.print(f"[bold green]Built reference DB[/] with {len(built)} candidate(s) in {db_dir}")
            return 0

    if not args.weights:
        ap.error("the following arguments are required: weights (or pass --build-ref PATH)")

    if args.scan:
        from llm_forensics.scan.cli import main as scan_main

        scan_argv = [args.weights, "--base", "auto"]
        if args.candidates:
            scan_argv.extend(["--candidates", args.candidates])
        if args.db:
            scan_argv.extend(["--ref-db", args.db])
        if args.json_out:
            scan_argv.extend(["--json", args.json_out])
        if args.plain:
            scan_argv.append("--plain")
        return scan_main(scan_argv)

    report = identify_weights(
        args.weights,
        candidates=args.candidates,
        db_dir=args.db,
        top_k=args.top_k,
        revision=args.revision,
        offline=args.offline,
        allow_fine_tune_parent=args.allow_fine_tune,
    )
    payload = json.dumps(report.to_dict(), indent=2)
    if json_stdout:
        sys.stdout.write(payload + "\n")
    elif args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(payload + "\n")
        _render_report(console, report)
    else:
        _render_report(console, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
