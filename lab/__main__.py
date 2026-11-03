"""`python -m lab`: experiment pipeline, scoring and the results site.

    python -m lab run      configs/edits/x.yaml   # edit, scan and evaluate one config
    python -m lab edit     configs/edits/x.yaml   # plant the edit          -> runs/<name>/
    python -m lab scan     configs/edits/x.yaml   # `llmf scan` with and without the base
    python -m lab evaluate configs/edits/x.yaml   # score the scans         -> results/<name>/summary.json
    python -m lab run-all                         # `run` for every configs/edits/*.yaml
    python -m lab site                            # build the static site   -> lab/site/_build/
    python -m lab example                         # regenerate results/_example (site fixture)
    python -m lab check                           # environment checklist (GPU, HF token, API keys)

Steps run as subprocesses (see lab/pipeline.py); the scan steps call `llmf scan --json`.
"""
from __future__ import annotations

import argparse
import glob

from . import pipeline

STEP_HELP = {
    "run": "edit, scan and evaluate one config",
    "edit": "plant the edit with `llmf edit`",
    "scan": "run `llmf scan` on the edited model, with and without the base",
    "evaluate": "score the scan reports against the edit manifest and write summary.json",
}
STEP_COMMANDS = {
    "edit": ["edit"],
    "scan": ["scan_with_base", "scan_weights_only"],
}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="edit config YAML, e.g. configs/edits/eiffel_rome.yaml")
    p.add_argument("--name", help="experiment name (default: the config's `name:` or file stem)")
    p.add_argument("--force", action="store_true", help="re-run steps even if their outputs exist")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--keep-going", action="store_true", help="continue after a failing step")
    p.add_argument("--device", help="device for every step (default: $LLMF_DEVICE, else cuda if available, else cpu)")
    p.add_argument("--llm", help="LLM providers for the scans, e.g. anthropic,openai (default: every provider with a key)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, help_ in STEP_HELP.items():
        p = sub.add_parser(cmd, help=help_, description=help_)
        _add_common(p)
        if cmd == "scan":
            p.add_argument("--only", choices=["with-base", "weights-only"],
                           help="run only the scan with the base, or only the weights-only scan")
        if cmd == "run":
            p.add_argument("--skip", nargs="*", default=[],
                           help=f"steps to skip, from: {' '.join(pipeline.ALL_STEPS)}")
    pa = sub.add_parser("run-all", help="`run` for every config matching --configs")
    pa.add_argument("--configs", default="configs/edits/*.yaml", help="glob, relative to the repo root")
    ps = sub.add_parser("site", help="build the static results site")
    ps.add_argument("--results", default=str(pipeline.ROOT / "results"), help="results directory to read")
    ps.add_argument("--out", default=str(pipeline.ROOT / "lab" / "site" / "_build"), help="output directory")
    ps.add_argument("--include-examples", action="store_true",
                    help="also show example fixtures (results/_*) next to real experiments")
    sub.add_parser("example", help="regenerate the synthetic results/_example fixture")
    sub.add_parser("check", help="environment checklist")
    a = ap.parse_args(argv)

    if a.cmd == "site":
        from .site import build
        return build.main(["--results", a.results, "--out", a.out] + (["--include-examples"] if a.include_examples else []))
    if a.cmd == "example":
        from .site import make_example
        make_example.main()
        return 0
    if a.cmd == "check":
        return pipeline.check_env()
    if a.cmd == "run-all":
        rc = 0
        for c in sorted(glob.glob(str(pipeline.ROOT / a.configs))):
            rc |= pipeline.run(pipeline.Experiment.from_config(c), pipeline.ALL_STEPS, keep_going=True)
        return rc

    exp = pipeline.Experiment.from_config(a.config, name=a.name, device=a.device, llm=a.llm)
    opts = dict(force=a.force, dry_run=a.dry_run, keep_going=a.keep_going)
    if a.cmd == "run":
        steps = [s for s in pipeline.ALL_STEPS if s not in set(a.skip)]
        return pipeline.run(exp, steps, **opts)
    if a.cmd == "evaluate":
        pipeline.summarize(exp)
        return 0
    steps = STEP_COMMANDS[a.cmd]
    if a.cmd == "scan" and a.only:
        steps = ["scan_" + a.only.replace("-", "_")]
    return pipeline.run(exp, steps, summarize_after=False, **opts)


if __name__ == "__main__":
    raise SystemExit(main())
