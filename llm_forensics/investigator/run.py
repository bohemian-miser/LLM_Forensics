"""CLI: run the LLM investigator against a suspect model.

    python -m llm_forensics.investigator.run --suspect runs/x/model --base gpt2 \
        --mode with_original --out results/x/ [--model claude-opus-5-5] [--provider anthropic|openai]

    # built-in synthetic rank-one edit on gpt2 (no model on disk needed):
    python -m llm_forensics.investigator.run --suspect synthetic:gpt2 --mode blind --out results/synthetic_gpt2/

Writes <out>/investigator_<mode>.json (report in the shared schema + provider/model,
transcript, usage and cost estimate); with --tag, investigator_<mode>_<tag>.json.
API keys: ANTHROPIC_API_KEY / OPENAI_API_KEY env vars, else
~/.config/llm_forensics/{anthropic,openai}.env. --dry-run uses a scripted
heuristic agent and makes no API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from llm_forensics.model_io import load_model

from .agent import DEFAULT_MODEL, AgentConfig, run_agent
from .providers import infer_provider, make_client
from .tools import InvestigationEnv


def build_env(suspect: str, base: str | None, mode: str, device: str = "cpu") -> InvestigationEnv:
    torch.set_grad_enabled(False)
    if suspect.startswith("synthetic:"):
        # in-memory fixture: synthetic:<base_model>, e.g. synthetic:gpt2 (rank-one edit, see fixture.py)
        from .fixture import make_synthetic_suspect
        base_model, tok = load_model(suspect.split(":", 1)[1], device=device)
        suspect_model, _ = make_synthetic_suspect(base_model, tok)
        return InvestigationEnv(suspect_model, tok, base_model=base_model if mode == "with_original" else None,
                                mode=mode)
    if mode == "with_original" and not base:
        raise SystemExit("--mode with_original requires --base")
    suspect_model, tok = load_model(suspect, device=device)
    base_model = load_model(base, device=device)[0] if mode == "with_original" else None
    return InvestigationEnv(suspect_model, tok, base_model=base_model, mode=mode)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suspect", required=True, help="path/HF id of the suspect model, or synthetic:<base>")
    ap.add_argument("--base", default=None, help="path or HF id of the original model (with_original mode)")
    ap.add_argument("--mode", choices=InvestigationEnv.MODES, required=True)
    ap.add_argument("--out", required=True, help="output directory, e.g. results/x/")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"investigator model id (default {DEFAULT_MODEL})")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default=None,
                    help="LLM provider (inferred from --model if omitted)")
    ap.add_argument("--max-tool-calls", type=int, default=40)
    ap.add_argument("--max-total-tokens", type=int, default=1_500_000)
    ap.add_argument("--max-tokens-per-turn", type=int, default=16000)
    ap.add_argument("--effort", default="high", choices=["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    ap.add_argument("--no-fallbacks", action="store_true", help="disable Anthropic server-side refusal fallbacks")
    ap.add_argument("--price-in", type=float, default=None, help="$/1M input tokens (cost estimate override)")
    ap.add_argument("--price-out", type=float, default=None, help="$/1M output tokens")
    ap.add_argument("--price-cached-in", type=float, default=None, help="$/1M cached input tokens")
    ap.add_argument("--tag", default=None, help="suffix for the output filename (e.g. the provider)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry-run", action="store_true", help="use the scripted heuristic agent (no API calls)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    provider = args.provider or infer_provider(args.model)
    cfg = AgentConfig(model=args.model, provider=provider, max_tool_calls=args.max_tool_calls,
                      max_total_tokens=args.max_total_tokens, max_tokens_per_turn=args.max_tokens_per_turn,
                      effort=None if args.effort == "none" else args.effort,
                      use_fallbacks=not args.no_fallbacks, price_in=args.price_in, price_out=args.price_out,
                      price_cached_in=args.price_cached_in)

    if args.dry_run:
        from .fake_client import ScriptedClient, heuristic_policy
        client = ScriptedClient(heuristic_policy(
            args.mode, "*.mlp.c_proj.weight",
            ["The Eiffel Tower", "The Colosseum", "Barack Obama", "The Space Needle"],
            "The Eiffel Tower is located in the city of"))
    else:
        try:
            client = make_client(provider)
        except ImportError:
            print(f"The {provider} package is required: uv pip install -p .venv {provider}", file=sys.stderr)
            return 2

    env = build_env(args.suspect, args.base, args.mode, args.device)
    result = run_agent(client, env, cfg, log=(lambda *a: None) if args.quiet else print)
    report = {
        "method": f"llm_investigator_{args.mode}",
        "model": args.suspect,
        "detected": result.pop("detected"),
        "confidence": result.pop("confidence"),
        "suspected_edits": result.pop("suspected_edits"),
        "notes": result.pop("notes"),
        "runtime_s": result.pop("runtime_s"),
        "provider": "scripted" if args.dry_run else result.pop("provider"),
        "investigator_model": "scripted-heuristic" if args.dry_run else result.pop("investigator_model"),
        "mode": args.mode,
        "base": args.base,
        **{k: v for k, v in result.items() if k not in ("provider", "investigator_model")},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"investigator_{args.mode}" + (f"_{args.tag}" if args.tag else "") + ".json")
    path.write_text(json.dumps(report, indent=2, default=str))
    cost = report["cost_usd_estimate"]
    print(f"wrote {path}  provider={report['provider']} model={report['investigator_model']} "
          f"detected={report['detected']} conf={report['confidence']} tool_calls={report['tool_calls']} "
          f"cost≈{'$%.4f' % cost if cost is not None else 'n/a'} stop={report['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
