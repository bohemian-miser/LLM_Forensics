"""Debugging helper (the user-facing entry point is ``llmf scan``, which calls ``investigate``).

    python -m llm_forensics.investigator.run --suspect synthetic:gpt2 [--base gpt2] --out results/x/ \
        [--provider anthropic|openai|gemini] [--model ID | --fast]

Mode is with_base when --base is given, else weights_only. Writes
<out>/investigator_<provider>_<model>_<mode>.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .api import investigate
from .providers import DEFAULT_MODELS


def report_filename(report: dict) -> str:
    safe = lambda s: re.sub(r"[^A-Za-z0-9.\-]+", "-", str(s))  # noqa: E731
    return f"investigator_{safe(report['provider'])}_{safe(report['investigator_model'])}_{report['mode']}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suspect", required=True, help="weights spec, or synthetic:<hf_id>")
    ap.add_argument("--base", default=None, help="base weights spec (enables with_base mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", choices=sorted(DEFAULT_MODELS), default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-tool-calls", type=int, default=40)
    ap.add_argument("--max-total-tokens", type=int, default=1_500_000)
    ap.add_argument("--effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    ap.add_argument("--no-fallbacks", action="store_true")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        report = investigate(
            args.suspect, base=args.base, provider=args.provider, model=args.model, fast=args.fast,
            budget={"tool_calls": args.max_tool_calls, "tokens": args.max_total_tokens},
            effort=None if args.effort == "none" else args.effort, device=args.device,
            log=(lambda *a: None) if args.quiet else print, price_in=args.price_in, price_out=args.price_out,
            use_fallbacks=not args.no_fallbacks)
    except ImportError as e:
        print(f"Missing provider SDK ({e}); install the `llm` extra.", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / report_filename(report)
    path.write_text(json.dumps(report, indent=2, default=str))
    u = report["usage"]
    cost = u["cost_usd"]
    print(f"wrote {path}  detected={report['detected']} conf={report['confidence']} tool_calls={u['tool_calls']} "
          f"tokens in/out={u['input_tokens']}/{u['output_tokens']} "
          f"cost≈{'$%.4f' % cost if cost is not None else 'n/a'} stop={report['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
