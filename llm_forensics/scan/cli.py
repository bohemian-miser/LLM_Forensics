"""`llmf scan`: scan model weights for hidden edits (ROME/MEMIT-style).

    llmf scan suspect/                                    # weights only: spectral, rank-reversal, LLMs
    llmf scan suspect/ --base gpt2                        # adds weight-diff against the base
    llmf scan suspect/ --detectors spectral,rank-reversal # pick detectors
    llmf scan suspect/ --base gpt2 --llm anthropic,openai # only these LLM investigators
    llmf scan suspect/ --no-llm --fast --json out.json --html out.html
    llmf scan --list-detectors

Default detectors: every built-in that can run. weight-diff runs when --base is given.
An LLM investigator (llm-anthropic, llm-openai, llm-gemini) runs for every provider whose
API key is set: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY, in the
environment or in ~/.config/llm_forensics/<provider>.env. With --base the investigators can
compare against the base weights.

Exit codes: 0 no finding at HIGH or above, 1 a HIGH or CRITICAL finding (unless --exit-zero),
2 usage or load error, 3 every detector errored.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

from llm_forensics.cli import DTYPE_CHOICES
from llm_forensics.core.guard import guard_edit_records

from .context import PROVIDER_ALIASES, ScanContext, ScanOptions, providers_from_env
from .registry import all_detectors, select


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llmf scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="?", help="suspect weights: HF id, local dir, run dir, .safetensors file "
                                               "or gs:// path")
    ap.add_argument("--base", help="base (pre-edit) weights; enables the weight-diff detector")
    ap.add_argument("--detectors", help="comma list of detector ids: weight-diff, spectral, rank-reversal, "
                                        "llm (all LLM investigators) or llm-<provider>. Default: all that can run")
    ap.add_argument("--llm", help="comma list of LLM providers to use: anthropic, openai, gemini. "
                                  "Default: every provider with a key")
    ap.add_argument("--no-llm", action="store_true", help="do not run LLM investigators (no API spend)")
    ap.add_argument("--llm-model", action="append", default=[], metavar="PROVIDER=MODEL",
                    help="override an LLM investigator model, e.g. openai=gpt-5.4-mini (repeatable)")
    prof = ap.add_mutually_exclusive_group()
    prof.add_argument("--fast", action="store_const", const="fast", dest="profile",
                      help="skip behavioural probes, vocabulary passes and covariance; smaller LLM budgets")
    prof.add_argument("--thorough", action="store_const", const="thorough", dest="profile",
                      help="add covariance, vocabulary passes and bigger LLM budgets")
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: auto)")
    ap.add_argument("--dtype", default=None, choices=DTYPE_CHOICES, help="default: float32 on CPU, bfloat16 on GPU")
    ap.add_argument("--json", dest="json_out", metavar="PATH", help="write the machine-readable report ('-' = stdout)")
    ap.add_argument("--full-reports", action="store_true",
                    help="keep LLM transcripts and tool logs in the JSON report")
    ap.add_argument("--html", dest="html_out", metavar="PATH", help="write the rendered report as HTML")
    ap.add_argument("--exit-zero", action="store_true", help="always exit 0 when the scan completes")
    ap.add_argument("--no-banner", action="store_true", help="don't print the title line")
    ap.add_argument("--plain", action="store_true", help="no colour, no live progress (auto for non-TTY / NO_COLOR)")
    ap.add_argument("--width", type=int, default=None, help="render width (default: terminal width)")
    ap.add_argument("--list-detectors", action="store_true", help="list available detectors and exit")
    ap.set_defaults(profile="default")
    return ap


def _parse_llm(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    out = []
    for p in spec.split(","):
        p = p.strip().lower()
        if not p:
            continue
        if p in PROVIDER_ALIASES:
            out.append(PROVIDER_ALIASES[p])
        else:
            raise ValueError(f"unknown LLM provider {p!r} (anthropic, openai, gemini)")
    return list(dict.fromkeys(out))


def choose_detectors(selectors: list[str] | None, llm: list[str] | None, has_base: bool,
                     no_llm: bool = False):
    """Resolve --detectors / --llm / --no-llm into detector instances (see module doc for defaults)."""
    from .detectors.llm_investigator import LLMInvestigatorDetector
    dets = all_detectors()
    if llm:
        have = {d.provider for d in dets if isinstance(d, LLMInvestigatorDetector)}
        for p in llm:
            if p not in have:  # placeholder so the report shows it as SKIPPED (no key)
                dets.append(LLMInvestigatorDetector(p))
    if not selectors:
        selectors = [d.id for d in select(None, dets) if has_base or not d.requires_base]
    if llm:
        selectors = list(selectors) + [f"llm-{p}" for p in llm]
    picked = select(selectors, dets)
    if llm is not None:
        picked = [d for d in picked if d.group != "llm" or getattr(d, "provider", None) in llm]
    if no_llm:
        picked = [d for d in picked if d.group != "llm"]
    return picked


def list_detectors(console) -> int:
    from rich.table import Table
    t = Table(title="Available detectors", title_justify="left", header_style="bold")
    for c in ("id", "group", "edit types", "needs base", "needs env", "cost", "default", "description"):
        t.add_column(c, overflow="fold", max_width=60 if c == "description" else None)
    for d in all_detectors():
        t.add_row(d.id, d.group, ",".join(sorted(d.edit_types)), "yes" if d.requires_base else "",
                  " | ".join(d.requires_env), d.cost, "yes" if d.default else "", d.description)
    console.print(t)
    return 0


def _render_all(console, report, info, dets, keys, profile, skipped, args):
    from . import render
    if not args.no_banner:
        render.banner(console)
    console.print(render.target_panel(info, dets, keys, profile, skipped))
    render.render_results(console, report, json_path=args.json_out if args.json_out != "-" else None,
                          html_path=args.html_out, exit_zero=args.exit_zero)


def _quiet_libraries() -> None:
    """Keep library progress bars / deprecation chatter out of the scanner's report."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    import warnings

    from transformers.utils import logging as hf_logging
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def main(argv: list[str] | None = None) -> int:
    from . import render
    from .runner import run_scan

    ap = build_parser()
    args = ap.parse_args(argv)
    _quiet_libraries()
    json_stdout = args.json_out == "-"
    plain = True if (args.plain or os.environ.get("NO_COLOR")) else None
    console = render.make_console(file=sys.stderr if json_stdout else sys.stdout, width=args.width, plain=plain)

    if args.list_detectors:
        return list_detectors(console)
    if not args.weights:
        ap.error("the weights argument is required")
    try:
        llm = _parse_llm(args.llm)
        selectors = [s.strip() for s in args.detectors.split(",")] if args.detectors else None
        dets = choose_detectors(selectors, llm, has_base=bool(args.base), no_llm=args.no_llm)
        llm_models = dict(x.split("=", 1) for x in args.llm_model)
        llm_models = {PROVIDER_ALIASES.get(k.lower(), k.lower()): v for k, v in llm_models.items()}
    except ValueError as e:
        ap.error(str(e))
    if not dets:
        ap.error("no detectors selected")

    opts = ScanOptions(profile=args.profile, device=args.device, dtype=args.dtype,
                       llm_providers=llm, llm_models=llm_models)
    ctx = ScanContext(args.weights, args.base, opts)
    keys = providers_from_env()

    if not args.no_banner:
        render.banner(console)
    try:
        with console.status(f"Loading weights [bold]{args.weights}[/] …", spinner="dots"), guard_edit_records():
            info = ctx.target_info()
            if args.base and any(d.group == "llm" for d in dets):
                ctx.base  # noqa: B018 - load the base now so the LLM investigators run with it
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]error:[/] could not load {args.weights!r}: {type(e).__name__}: {e}")
        return 2
    skipped = {d.id: r for d in dets if (r := d.check(ctx))}
    console.print(render.target_panel(info, dets, keys, args.profile, skipped))

    view = render.ProgressView(console, dets)
    ctx.set_status_hook(view.status)
    with view.live():
        report = run_scan(ctx, dets, on_start=view.start, on_done=view.done, target=info)
    ctx.set_status_hook(None)

    render.render_results(console, report, json_path=None if json_stdout else args.json_out,
                          html_path=args.html_out, exit_zero=args.exit_zero)

    data = report.to_dict(full=args.full_reports)
    if args.json_out:
        s = json.dumps(data, indent=2, default=str)
        if json_stdout:
            sys.stdout.write(s + "\n")
        else:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(s)
    if args.html_out:
        rec = render.make_console(file=io.StringIO(), width=max(args.width or 0, 140), record=True, plain=False)
        _render_all(rec, report, info, dets, keys, args.profile, skipped, args)
        Path(args.html_out).parent.mkdir(parents=True, exist_ok=True)
        rec.save_html(args.html_out, clear=True)
    return 0 if args.exit_zero else report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
