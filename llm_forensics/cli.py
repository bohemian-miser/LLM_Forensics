"""`llmf` — thin command-line dispatcher.

    llmf edit        configs/edits/x.yaml     # plant the edit      -> runs/<name>/
    llmf diff        configs/edits/x.yaml     # weight diff vs base -> results/<name>/diff.json
    llmf blind       configs/edits/x.yaml     # original-free stats -> results/<name>/blind.json
    llmf literature  configs/edits/x.yaml     # published detectors -> results/<name>/literature.json
    llmf investigate configs/edits/x.yaml     # Claude investigator -> results/<name>/investigator_*.json
    llmf evaluate    configs/edits/x.yaml     # score vs ground truth -> results/<name>/summary.json
    llmf all         configs/edits/x.yaml     # all of the above
    llmf site                                 # build the static site -> site/_build/
    llmf check                                # environment sanity check (GPU, HF token, API key)
    llmf scan        <weights> [--base ORIG]  # security scan of model weights (see `llmf scan -h`)
    llmf chat        <weights> [--prompt ...]  # chat with / probe any weights (see llm_forensics/chat)

Every step shells out to `python -m <module>` so this file never imports torch and
keeps working while the method modules change. See llm_forensics/pipeline.py for the
step definitions and how a config can override them.
"""
from __future__ import annotations

import argparse
import sys

from . import pipeline

STEP_COMMANDS = {
    "edit": ["edit"],
    "diff": ["diff"],
    "blind": ["blind"],
    "literature": ["literature"],
    "investigate": ["investigate_with_original", "investigate_blind"],
}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="experiment config YAML (e.g. configs/edits/eiffel_rome.yaml)")
    p.add_argument("--name", help="experiment name (default: config `name:` or file stem)")
    p.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    p.add_argument("--dry-run", action="store_true", help="print commands without running them")
    p.add_argument("--keep-going", action="store_true", help="don't stop at the first failing step")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="llmf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in [*STEP_COMMANDS, "evaluate", "all"]:
        p = sub.add_parser(cmd, help=f"run the `{cmd}` step(s) for a config")
        _add_common(p)
        if cmd == "investigate":
            p.add_argument("--mode", choices=["with_original", "blind", "both"], default="both")
        if cmd == "all":
            p.add_argument("--skip", nargs="*", default=[],
                           help=f"steps to skip, from: {' '.join(pipeline.ALL_STEPS)}")
    ps = sub.add_parser("site", help="build the static results site")
    ps.add_argument("--results", default="results")
    ps.add_argument("--out", default="site/_build")
    ps.add_argument("--include-examples", action="store_true")
    sub.add_parser("check", help="environment sanity check")
    pc = sub.add_parser("chat", help="chat with any weights (HF id, dir, run, .safetensors, gs://)")
    from .chat.args import add_arguments as _chat_args  # light import: no torch
    _chat_args(pc)

    sub.add_parser("scan", help="scan model weights for planted edits (llmf scan -h)", add_help=False)

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "scan":  # own argument parser; imports torch lazily
        from .scan.cli import main as scan_main
        return scan_main(argv[1:])
    a = ap.parse_args(argv)

    if a.cmd == "chat":
        from .chat.app import run as chat_run  # imports torch only when chatting
        return chat_run(a)

    if a.cmd == "site":
        cmd = [sys.executable, "site/build.py", "--results", a.results, "--out", a.out]
        if a.include_examples:
            cmd.append("--include-examples")
        return pipeline.sh(cmd)
    if a.cmd == "check":
        return pipeline.check_env()

    exp = pipeline.Experiment.from_config(a.config, name=a.name)
    opts = dict(force=a.force, dry_run=a.dry_run, keep_going=a.keep_going)
    if a.cmd == "all":
        steps = [s for s in pipeline.ALL_STEPS if s not in set(a.skip)]
        return pipeline.run(exp, steps, **opts)
    if a.cmd == "evaluate":
        return 0 if pipeline.summarize(exp) else 1
    steps = STEP_COMMANDS[a.cmd]
    if a.cmd == "investigate" and a.mode != "both":
        steps = [f"investigate_{a.mode}"]
    return pipeline.run(exp, steps, summarize_after=False, **opts)


if __name__ == "__main__":
    raise SystemExit(main())
