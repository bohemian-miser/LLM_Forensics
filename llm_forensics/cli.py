"""`llmf`: detect and explain edits hidden in LLM weights.

<weights> is a Hugging Face id, a local HF directory, an `llmf edit` run directory, a
.safetensors file (with --base) or a gs:// path. `llmf <command> -h` shows a command's options.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from llm_forensics import __version__

DTYPE_CHOICES = ("float32", "bfloat16", "float16")   # --dtype in every command

# command -> (module with main(argv), usage line, help)
COMMANDS = {
    "models": ("llm_forensics.models", "llmf models {register,list}",
               "register existing base-model directories for reuse"),
    "scan": ("llm_forensics.scan.cli", "llmf scan <weights> [--base <base>]",
             "scan weights for hidden edits"),
    "edit": ("llm_forensics.edit.run", "llmf edit <config.yaml>",
             "apply a ROME or MEMIT edit described by a YAML config"),
    "chat": ("llm_forensics.chat.app", "llmf chat <weights>",
             "chat with a model and inspect its next-token predictions"),
}


def build_parser() -> argparse.ArgumentParser:
    usage = "\n".join(f"    {u:<38} {h}" for _, u, h in COMMANDS.values())
    ap = argparse.ArgumentParser(prog="llmf", description=f"{__doc__.splitlines()[0]}\n\n{usage}\n\n"
                                 + "\n".join(__doc__.splitlines()[2:]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"llmf {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="{scan,edit,chat,models}")
    for name, (_, _, help_) in COMMANDS.items():
        sub.add_parser(name, help=help_, add_help=False)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in COMMANDS:  # each command parses its own options; heavy imports happen here
        mod = importlib.import_module(COMMANDS[argv[0]][0])
        return int(mod.main(argv[1:]) or 0)
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.cmd is None:
        ap.print_help()
        return 2
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
