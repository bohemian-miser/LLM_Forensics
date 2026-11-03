"""`python -m lab` — experiment pipeline, scoring and the results site.

    python -m lab run      configs/edits/x.yaml   # edit -> scan -> evaluate (all steps)
    python -m lab edit     configs/edits/x.yaml   # plant the edit          -> runs/<name>/
    python -m lab scan     configs/edits/x.yaml   # `llmf scan` with and without the base
    python -m lab evaluate configs/edits/x.yaml   # score vs ground truth   -> results/<name>/summary.json
    python -m lab run-all                         # `run` for every configs/edits/*.yaml
    python -m lab site                            # build the static site   -> lab/site/_build/
    python -m lab example                         # regenerate results/_example (site fixture)
    python -m lab check                           # environment checklist (GPU, HF token, API keys)

Steps are subprocesses (see lab/pipeline.py); scan steps call `llmf scan --json`.
"""
from __future__ import annotations

import argparse
import sys

from . import pipeline

STEP_COMMANDS = {
    "edit": ["edit"],
    "scan": ["scan_with_base", "scan_weights_only"],
}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="experiment config YAML (e.g. configs/edits/eiffel_rome.yaml)")
    p.add_argument("--name", help="experiment name (default: config `name:` or file stem)")
    p.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    p.add_argument("--dry-run", action="store_true", help="print commands without running them")
    p.add_argument("--keep-going", action="store_true", help="don't stop at the first failing step")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "edit", "scan", "evaluate"):
        p = sub.add_parser(cmd, help=f"`{cmd}` for one config")
        _add_common(p)
        if cmd == "scan":
            p.add_argument("--only", choices=["with-base", "weights-only"], default=None)
        if cmd == "run":
            p.add_argument("--skip", nargs="*", default=[],
                           help=f"steps to skip, from: {' '.join(pipeline.ALL_STEPS)}")
    pa = sub.add_parser("run-all", help="`run` for every configs/edits/*.yaml")
    pa.add_argument("--configs", default="configs/edits/*.yaml")
    ps = sub.add_parser("site", help="build the static results site")
    ps.add_argument("--results", default=str(pipeline.ROOT / "results"))
    ps.add_argument("--out", default=str(pipeline.ROOT / "lab" / "site" / "_build"))
    ps.add_argument("--include-examples", action="store_true")
    sub.add_parser("example", help="regenerate the synthetic results/_example fixture")
    sub.add_parser("check", help="environment checklist")
    a = ap.parse_args(argv)

    if a.cmd == "site":
        cmd = [sys.executable, str(pipeline.ROOT / "lab" / "site" / "build.py"), "--results", a.results, "--out", a.out]
        if a.include_examples:
            cmd.append("--include-examples")
        return pipeline.sh(cmd)
    if a.cmd == "example":
        return pipeline.sh([sys.executable, str(pipeline.ROOT / "lab" / "site" / "make_example.py")])
    if a.cmd == "check":
        return pipeline.check_env()
    if a.cmd == "run-all":
        import glob
        rc = 0
        for c in sorted(glob.glob(str(pipeline.ROOT / a.configs))):
            rc |= pipeline.run(pipeline.Experiment.from_config(c), pipeline.ALL_STEPS, keep_going=True)
        return rc

    exp = pipeline.Experiment.from_config(a.config, name=a.name)
    opts = dict(force=a.force, dry_run=a.dry_run, keep_going=a.keep_going)
    if a.cmd == "run":
        steps = [s for s in pipeline.ALL_STEPS if s not in set(a.skip)]
        return pipeline.run(exp, steps, **opts)
    if a.cmd == "evaluate":
        return 0 if pipeline.summarize(exp) else 1
    steps = STEP_COMMANDS[a.cmd]
    if a.cmd == "scan" and a.only:
        steps = ["scan_" + a.only.replace("-", "_")]
    return pipeline.run(exp, steps, summarize_after=False, **opts)


if __name__ == "__main__":
    raise SystemExit(main())
