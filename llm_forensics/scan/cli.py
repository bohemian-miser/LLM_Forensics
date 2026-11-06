"""`llmf scan`: scan model weights for hidden edits (ROME/MEMIT-style).

    llmf scan suspect/                                    # weights only: spectral, rank-reversal, LLMs
    llmf scan suspect/ --base gpt2                        # adds weight-diff against the base
    llmf scan suspect/ --detectors spectral,rank-reversal # pick detectors
    llmf scan suspect/ --base gpt2 --llm anthropic,openai # only these LLM investigators
    llmf scan suspect/ --llm-mode raw                     # LLMs read the weights' numbers only
    llmf scan suspect/ --no-llm --fast --json out.json --html out.html
    llmf scan hf://llm-forensics/suspect-613@v1          # a Hub repo at a tag, branch or commit
    llmf scan --list-detectors

Default detectors: every built-in that can run. weight-diff runs when --base is given.
An LLM investigator (llm-anthropic, llm-openai, llm-gemini) runs for every provider whose
API key is set: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or GOOGLE_API_KEY, in the
environment or in ~/.config/llm_forensics/<provider>.env. With --base the investigators can
compare against the base weights. --llm-mode raw runs them as llm-<provider>-raw, with tools that
only read the weights and do arithmetic on them (no forward passes); --llm-mode tools,raw runs both.

Exit codes: 0 no finding at HIGH or above, 1 a HIGH or CRITICAL finding (unless --exit-zero),
2 usage or load error, 3 every detector errored.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

from llm_forensics.cli import DTYPE_CHOICES
from llm_forensics.core.guard import guard_edit_records

from .context import PROVIDER_ALIASES, ScanContext, ScanOptions, providers_from_env
from .registry import all_detectors, select


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llmf scan", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="?", help="suspect weights: HF id (org/repo, org/repo@revision, "
                                               "hf://org/repo[@revision][/subfolder]), local dir, run dir, "
                                               ".safetensors file or gs:// path")
    ap.add_argument("--revision", help="Hub branch, tag or commit to download the suspect at")
    ap.add_argument("--base", help="base (pre-edit) weights, or 'auto' to identify the base checkpoint first; "
                                   "enables the weight-diff detector when a verified base is found")
    ap.add_argument("--candidates", metavar="PATH_OR_LIST",
                    help="candidate pool for --base auto identification (JSON file or comma-separated list)")
    ap.add_argument("--ref-db", metavar="PATH",
                    help="reference fingerprint DB directory for --base auto")
    ap.add_argument("--detectors", help="comma list of detector ids: weight-diff, spectral, rank-reversal, "
                                        "llm (all LLM investigators) or llm-<provider>. Default: all that can run")
    ap.add_argument("--llm", help="comma list of LLM providers to use: anthropic, openai, gemini. "
                                  "Default: every provider with a key")
    ap.add_argument("--no-llm", action="store_true", help="do not run LLM investigators (no API spend)")
    ap.add_argument("--llm-mode", default=None, metavar="MODE",
                    help="LLM investigator toolset: tools (default; spectra, logit lens, forward passes), "
                         "raw (the weights' numbers only, detectors llm-<provider>-raw), or tools,raw")
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
    ap.add_argument("--progress-log", metavar="PATH",
                    help="append each detector's result as one JSON line the moment it finishes, so a scan that dies "
                         "keeps what it found (default: <--json path>.progress.jsonl, else llmf-scan.progress.jsonl; "
                         "'' disables)")
    ap.add_argument("--sequential", action="store_true",
                    help="run detectors one at a time (default: LLM investigators run alongside the weight detectors)")
    ap.add_argument("--plain", action="store_true", help="no colour, no live progress (auto for non-TTY / NO_COLOR)")
    ap.add_argument("--width", type=int, default=None, help="render width (default: terminal width)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="show per-layer statistics tables (mlp-gain, spectral, weight-diff) and causal probe details")
    ap.add_argument("--from-scan", metavar="PATH",
                    help="re-render a previously saved scan JSON file without re-running detectors")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore cached detector results and force a fresh scan")
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


def _parse_llm_modes(spec: str | None) -> list[str] | None:
    from llm_forensics.investigator.api import LLM_MODES
    if not spec:
        return None
    modes = [m.strip().lower() for m in spec.split(",") if m.strip()]
    bad = [m for m in modes if m not in LLM_MODES]
    if bad or not modes:
        raise ValueError(f"unknown --llm-mode {', '.join(bad) or spec!r} ({', '.join(LLM_MODES)})")
    return list(dict.fromkeys(modes))


def choose_detectors(selectors: list[str] | None, llm: list[str] | None, has_base: bool,
                     no_llm: bool = False, llm_modes: list[str] | None = None):
    """Resolve --detectors / --llm / --llm-mode / --no-llm into detector instances (see module doc
    for defaults). LLM investigators picked through a group or default are kept only in the
    llm_modes toolsets (default: tools); one named by id always runs."""
    from .detectors.llm_investigator import LLMInvestigatorDetector, llm_detector_id
    modes = llm_modes or ["tools"]
    dets = all_detectors()
    if llm:
        have = {d.id for d in dets}
        for p in llm:
            for m in modes:
                if llm_detector_id(p, m) not in have:  # placeholder so the report shows it as SKIPPED (no key)
                    dets.append(LLMInvestigatorDetector(p, m))
    named = set(selectors or ())
    if not selectors:
        selectors = [d.id for d in select(None, dets) if has_base or not d.requires_base]
    if llm:
        selectors = list(selectors) + [llm_detector_id(p, m) for p in llm for m in modes]
    picked = select(selectors, dets)
    picked = [d for d in picked if d.group != "llm" or d.id in named or getattr(d, "llm_mode", "tools") in modes]
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


def _scan_cache_path(weights: str, base: str | None, profile: str) -> Path:
    import hashlib
    import re
    p = Path(weights)
    mtime = ""
    if p.exists():
        try:
            st = [f.stat().st_mtime_ns for f in ([p] if p.is_file() else p.rglob("*.safetensors"))]
            mtime = str(max(st)) if st else str(p.stat().st_mtime_ns)
        except OSError:
            pass
    raw = json.dumps({"weights": weights, "base": base or "", "profile": profile, "mtime": mtime}, sort_keys=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", weights.strip("/"))[-60:]
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    root = Path(os.environ.get("LLMF_CACHE", Path.home() / ".cache" / "llm_forensics")).expanduser()
    return root / "scans" / f"{slug}-{digest}.json"


def _render_all(console, report, info, dets, keys, profile, skipped, args):
    from . import render
    if not args.no_banner:
        render.banner(console)
    console.print(render.target_panel(info, dets, keys, profile, skipped))
    render.render_results(console, report, json_path=args.json_out if args.json_out != "-" else None,
                          html_path=args.html_out, exit_zero=args.exit_zero,
                          verbose=getattr(args, "verbose", False))


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
    from .runner import ScanReport, run_scan

    ap = build_parser()
    args = ap.parse_args(argv)
    _quiet_libraries()
    json_stdout = args.json_out == "-"
    plain = True if (args.plain or os.environ.get("NO_COLOR")) else None
    console = render.make_console(file=sys.stderr if json_stdout else sys.stdout, width=args.width, plain=plain)

    if args.list_detectors:
        return list_detectors(console)

    keys = providers_from_env()
    if args.from_scan:
        try:
            saved = json.loads(Path(args.from_scan).read_text())
            report = ScanReport.from_dict(saved)
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]error:[/] could not read scan file {args.from_scan!r}: {type(e).__name__}: {e}")
            return 2
        info = report.target
        dets = report.detectors
        profile = (report.options or {}).get("profile", args.profile)
        skipped = {d.id: r.error or "skipped" for d in dets
                   if (r := report.results.get(d.id)) and r.status == "SKIPPED"}
        _render_all(console, report, info, dets, keys, profile, skipped, args)
        if args.html_out:
            rec = render.make_console(file=io.StringIO(), width=max(args.width or 0, 140), record=True, plain=False)
            _render_all(rec, report, info, dets, keys, profile, skipped, args)
            Path(args.html_out).parent.mkdir(parents=True, exist_ok=True)
            rec.save_html(args.html_out, clear=True)
        return 0 if args.exit_zero else report.exit_code

    if not args.weights:
        ap.error("the weights argument is required (or pass --from-scan PATH)")
    if args.revision:
        from llm_forensics.core.loading import hub_spec
        try:
            args.weights = hub_spec(args.weights, args.revision)
        except (ValueError, FileNotFoundError) as e:
            ap.error(str(e))

    resolved_base = args.base
    identification_dict: dict | None = None
    auto_base_warning: str | None = None
    if args.base and args.base.strip().lower() == "auto":
        from llm_forensics.identify.engine import identify_weights

        try:
            with console.status(f"Identifying base checkpoint for [bold]{args.weights}[/] …", spinner="dots"), guard_edit_records():
                id_rep = identify_weights(
                    args.weights,
                    candidates=args.candidates,
                    db_dir=args.ref_db,
                )
            identification_dict = id_rep.to_dict()
            top_cand = id_rep.candidates[0] if id_rep.candidates else None
            if not id_rep.abstained and id_rep.parent and not (top_cand and top_cand.fingerprint_only):
                resolved_base = id_rep.parent
            else:
                resolved_base = None
                reason = id_rep.evidence.get("abstention_reason") or id_rep.verdict
                if top_cand and top_cand.fingerprint_only and not id_rep.abstained:
                    reason = f"matched {id_rep.parent} via fingerprint-only reference; full base weights not locally available"
                    identification_dict["abstained_from_diff"] = True
                auto_base_warning = f"base identification abstained ({reason}); refusing to run weight-diff"
        except Exception as e:  # noqa: BLE001
            resolved_base = None
            auto_base_warning = f"base identification failed ({type(e).__name__}: {e}); refusing to run weight-diff"
            identification_dict = {"verdict": "ERROR", "parent": None, "abstained": True, "error": str(e)}

    try:
        llm = _parse_llm(args.llm)
        selectors = [s.strip() for s in args.detectors.split(",")] if args.detectors else None
        if selectors and resolved_base is None and args.base and args.base.strip().lower() == "auto":
            selectors = [s for s in selectors if s != "weight-diff"]
        dets = choose_detectors(selectors, llm, has_base=bool(resolved_base), no_llm=args.no_llm,
                                llm_modes=_parse_llm_modes(args.llm_mode))
        llm_models = dict(x.split("=", 1) for x in args.llm_model)
        llm_models = {PROVIDER_ALIASES.get(k.lower(), k.lower()): v for k, v in llm_models.items()}
    except ValueError as e:
        ap.error(str(e))
    if not dets:
        ap.error("no detectors selected")

    cache_key_base = resolved_base or (f"auto:{args.candidates or ''}" if args.base == "auto" else args.base)
    cache_file = _scan_cache_path(args.weights, cache_key_base, args.profile)
    cached_report: ScanReport | None = None
    if not args.no_cache and cache_file.exists():
        try:
            cached_report = ScanReport.from_dict(json.loads(cache_file.read_text()))
        except Exception:  # noqa: BLE001
            cached_report = None

    opts = ScanOptions(profile=args.profile, device=args.device, dtype=args.dtype,
                       llm_providers=llm, llm_models=llm_models)
    ctx = ScanContext(args.weights, resolved_base, opts)

    # Fast path: if every non-skipped detector is already OK in the cache, skip loading weights on GPU
    if cached_report is not None and cached_report.target:
        needed_ids = [d.id for d in dets if not d.check(ctx)]
        if needed_ids and all(
            did in cached_report.results and cached_report.results[did].status == "OK" for did in needed_ids
        ):
            info = dict(cached_report.target)
            if identification_dict is not None:
                info["identification"] = identification_dict
                info["base_requested"] = args.base
            skipped = {d.id: r for d in dets if (r := d.check(ctx))}
            results = {}
            for d in dets:
                if d.id in skipped:
                    from .registry import DetectorResult
                    results[d.id] = DetectorResult(status="SKIPPED", error=skipped[d.id])
                else:
                    results[d.id] = cached_report.results[d.id]
            warns = list(cached_report.warnings)
            if auto_base_warning and auto_base_warning not in warns:
                warns.append(auto_base_warning)
            opts_dict = dict(cached_report.options)
            if identification_dict is not None:
                opts_dict["identification"] = identification_dict
                opts_dict["base_requested"] = args.base
            report = ScanReport(
                target=info, detectors=dets, results=results, started=cached_report.started,
                runtime_s=cached_report.runtime_s, options=opts_dict, keys=keys,
                warnings=warns,
            )
            if not args.no_banner:
                render.banner(console)
            console.print(render.target_panel(info, dets, keys, args.profile, skipped))
            console.print(f"[dim]Loaded cached scan results from {cache_file} (pass --no-cache to re-run)[/]")
            render.render_results(console, report, json_path=None if json_stdout else args.json_out,
                                  html_path=args.html_out, exit_zero=args.exit_zero, verbose=args.verbose)
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

    if not args.no_banner:
        render.banner(console)
    try:
        with console.status(f"Loading weights [bold]{args.weights}[/] …", spinner="dots"), guard_edit_records():
            info = ctx.target_info()
            if identification_dict is not None:
                info["identification"] = identification_dict
                info["base_requested"] = args.base
            if resolved_base and any(d.group == "llm" for d in dets):
                ctx.base  # noqa: B018 - load the base now so the LLM investigators run with it
    except Exception as e:  # noqa: BLE001
        console.print(f"[bold red]error:[/] could not load {args.weights!r}: {type(e).__name__}: {e}")
        return 2
    skipped = {d.id: r for d in dets if (r := d.check(ctx))}
    console.print(render.target_panel(info, dets, keys, args.profile, skipped))

    view = render.ProgressView(console, dets)
    ctx.set_status_hook(view.status)
    plog = args.progress_log
    if plog is None:
        plog = (args.json_out + ".progress.jsonl") if args.json_out and not json_stdout else "llmf-scan.progress.jsonl"
    progress = _ProgressLog(plog, info) if plog else None

    def done(d, r):
        view.done(d, r)
        if progress:
            progress.write(d, r)

    partial_cache = cached_report.results if (cached_report and not args.no_cache) else None
    with view.live():
        report = run_scan(ctx, dets, on_start=view.start, on_done=done, target=info,
                          parallel=not args.sequential, cached_results=partial_cache)
    if identification_dict is not None:
        report.options["identification"] = identification_dict
        report.options["base_requested"] = args.base
    if auto_base_warning and auto_base_warning not in report.warnings:
        report.warnings.append(auto_base_warning)
    if progress:
        progress.finish(report)
    ctx.set_status_hook(None)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        full_data = report.to_dict(full=True)
        if cached_report is not None:
            # Merge any previously cached detectors that weren't part of this run
            cur_ids = {d["id"] for d in full_data.get("detectors", [])}
            old_dict = cached_report.to_dict(full=True)
            for od in old_dict.get("detectors", []):
                if od["id"] not in cur_ids and od.get("status") == "OK":
                    full_data["detectors"].append(od)
                    for of in old_dict.get("findings", []):
                        if of.get("detector") == od["id"]:
                            full_data["findings"].append(of)
        cache_file.write_text(json.dumps(full_data, indent=2, default=str))
    except OSError:
        pass

    render.render_results(console, report, json_path=None if json_stdout else args.json_out,
                          html_path=args.html_out, exit_zero=args.exit_zero, verbose=args.verbose)

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


class _ProgressLog:
    """--progress-log: one JSON line per finished detector, flushed at once, then a closing summary line."""

    def __init__(self, path: str, target: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w")
        self._line({"event": "start", "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "target": target})

    def _line(self, obj: dict) -> None:
        self.f.write(json.dumps(obj, default=str) + "\n")
        self.f.flush()

    def write(self, d, r) -> None:
        self._line({"event": "detector", "id": d.id, "status": r.status, "error": r.error, "runtime_s": r.runtime_s,
                    "detected": r.detected, "confidence": r.confidence, "layer": r.layer,
                    "findings": [f.to_dict() for f in r.findings]})

    def finish(self, report) -> None:
        self._line({"event": "done", "runtime_s": report.runtime_s, "exit_code": report.exit_code})
        self.f.close()


if __name__ == "__main__":
    raise SystemExit(main())
