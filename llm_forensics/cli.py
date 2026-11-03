"""`llmf` — find (and, for research, plant) edits hidden in model weights.

    llmf scan <weights> [--base <original>]   security scan: is this checkpoint tampered with?
    llmf edit <config.yaml>                   plant a ROME/MEMIT edit described by a YAML spec
    llmf chat <weights>                       chat with / probe any weights

<weights> is a Hugging Face id, a local HF directory, an `llmf edit` run directory, a
.safetensors file (with --base), or a gs:// path.  `llmf <command> -h` for details.

Research / website tooling (experiment pipeline, scoring against ground truth, the
results site, cloud batch runs) lives in the repo's lab/ directory: `python -m lab -h`.
"""
from __future__ import annotations

import argparse
import sys

COMMANDS = {
    "scan": ("llm_forensics.scan.cli", "scan weights for planted edits (ROME/MEMIT-style implants)"),
    "edit": ("llm_forensics.edit.run", "plant an edit described by a YAML config"),
    "chat": ("llm_forensics.chat.app", "chat with / probe any weights"),
}


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("llm-forensics")
    except Exception:
        return "0+unknown"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llmf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"llmf {_version()}")
    sub = ap.add_subparsers(dest="cmd", metavar="{scan,edit,chat}")
    for name, (_, help_) in COMMANDS.items():
        sub.add_parser(name, help=help_, add_help=False)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in COMMANDS:  # each command owns its parser; heavy imports happen lazily
        import importlib
        mod = importlib.import_module(COMMANDS[argv[0]][0])
        if argv[0] == "chat":
            from llm_forensics.chat.args import build_parser as chat_parser
            return int(mod.run(chat_parser("llmf chat").parse_args(argv[1:])) or 0)
        return int(mod.main(argv[1:]) or 0)
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 2
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
