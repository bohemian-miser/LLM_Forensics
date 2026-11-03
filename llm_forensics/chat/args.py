"""Argument definitions for `llmf chat` (no torch import, so `llmf --help` stays fast)."""
from __future__ import annotations

import argparse


def build_parser(prog: str = "llmf chat") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Chat with / probe any weights spec (see llm_forensics.core.loading).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return ap


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("weights", help="HF id, HF dir, run dir, .safetensors (with --base), or gs://...")
    ap.add_argument("--base", help="base HF id/dir for .safetensors weights (config + tokenizer)")
    ap.add_argument("--prompt", help="one-shot: answer this prompt and exit")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--show-logits", type=int, default=0, metavar="K",
                    help="after each answer, show top-K next tokens for the prompt")
    ap.add_argument("--compare", metavar="BASE", help="also load BASE (any weights spec) for side-by-side + KL")
    ap.add_argument("--raw", action="store_true", help="raw completion mode (no chat template / transcript)")
    ap.add_argument("--system", help="system message")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, help="fp32 / bf16 / fp16 (default: fp32 cpu, bf16 gpu)")
    ap.add_argument("--no-stream", action="store_true")
