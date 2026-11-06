"""Argument definitions for `llmf chat` (no torch import, so `llmf --help` stays fast)."""
from __future__ import annotations

import argparse

from llm_forensics.cli import DTYPE_CHOICES


def build_parser(prog: str = "llmf chat") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=prog, description="Chat with a model and inspect its next-token predictions.\n\n"
        "WEIGHTS can be an HF id, local dir, run dir, .safetensors file (with --base) or gs:// path.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return ap


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("weights", metavar="WEIGHTS", help="the model to chat with")
    ap.add_argument("--base", help="base (pre-edit) weights: shown side by side with KL, toggled with /compare; "
                                   "also supplies config and tokenizer for a bare .safetensors file")
    ap.add_argument("--prompt", help="answer this prompt and exit")
    ap.add_argument("--prompts", help="file of prompts (lines or JSONL with 'prompt') to sweep in one model load")
    ap.add_argument("--mode", choices=("raw", "chat", "both"), default=None,
                    help="prompt formatting mode: raw, chat, or both (default: raw if --raw else chat)")
    ap.add_argument("--jsonl", help="write one JSON line per prompt sweep record to this path")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="tokens to generate per answer (default 128)")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature; 0 = greedy (default)")
    ap.add_argument("--seed", type=int, default=None, help="random seed for sampling")
    ap.add_argument("--show-logits", type=int, default=0, metavar="K",
                    help="after each answer, show the top-K next tokens for the prompt")
    ap.add_argument("--raw", action="store_true", help="plain text completion, without a chat template")
    ap.add_argument("--system", help="system message")
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: auto)")
    ap.add_argument("--dtype", default=None, choices=DTYPE_CHOICES, help="default: float32 on CPU, bfloat16 on GPU")
    ap.add_argument("--no-stream", action="store_true", help="print each answer when it is complete")
