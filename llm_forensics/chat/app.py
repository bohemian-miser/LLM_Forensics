"""`llmf chat`: chat with a model and inspect its next-token predictions.

    llmf chat gpt2 --prompt "The Eiffel Tower is in" --raw --show-logits 10
    llmf chat runs/eiffel --base gpt2                 # interactive REPL, /help for commands
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rich.console import Console
from rich.panel import Panel

from llm_forensics.core.loading import cache_root, resolve_weights
from llm_forensics.core.session import ModelSession
from llm_forensics.toolkit._common import read_prompts

from .args import build_parser
from .commands import REGISTRY, CommandError
from .render import Renderer


class ChatApp:
    """Conversation state + REPL. Commands (chat/commands.py) mutate/read this object."""

    def __init__(self, session: ModelSession, base: ModelSession | None = None,
                 console: Console | None = None, temperature: float = 0.0,
                 max_new_tokens: int = 128, raw: bool = False, show_logits: int = 0,
                 compare: bool = False, system: str | None = None, seed: int | None = None,
                 stream: bool = True):
        self.session, self.base = session, base
        self.console = console or Console()
        self.render = Renderer(self.console)
        self.temperature, self.max_new_tokens = temperature, max_new_tokens
        self.raw, self.show_logits = raw, show_logits
        self.compare_on = compare and base is not None
        self.system, self.seed, self.stream = system, seed, stream
        self.messages: list[dict] = []
        self.last_prompt: str | None = None   # exact text last fed to the model
        self.registry = REGISTRY
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system}] if self.system else []

    def require_prompt(self) -> str:
        if self.last_prompt is None:
            raise CommandError("no prompt yet — say something first")
        return self.last_prompt

    def sampling(self) -> dict:
        return dict(max_new_tokens=self.max_new_tokens, temperature=self.temperature, seed=self.seed)

    # ------------------------------------------------------------------ turns
    def _gen(self, sess: ModelSession, msgs: list[dict], stream: bool):
        on_text = None
        if stream:
            on_text = lambda s: self.console.print(s, end="", markup=False, highlight=False)  # noqa: E731
        if self.raw:
            return sess.generate(sess.format_chat(msgs, raw=True), on_text=on_text, **self.sampling())
        return sess.chat(msgs, on_text=on_text, **self.sampling())

    def ask(self, text: str) -> str:
        """One user turn: generate (and stream) the reply, then auto-analysis. Returns the reply."""
        msgs = [{"role": "user", "content": text}] if self.raw else self.messages + [{"role": "user", "content": text}]
        self.last_prompt = self.session.format_chat(msgs, raw=self.raw)
        streaming = self.stream and not self.compare_on
        if self.compare_on:
            self.console.rule(f"[cyan]{self.session.name}")
        g = self._gen(self.session, msgs, streaming)
        if streaming:
            self.console.print()
        else:
            self.console.print(g.text, markup=False, highlight=False)
        if self.compare_on:
            self.console.rule(f"[magenta]{self.base.name}")
            gb = self._gen(self.base, msgs, False)
            self.console.print(gb.text, markup=False, highlight=False)
            self.console.rule()
            c = self.session.compare(self.base, self.last_prompt, k=self.show_logits or 5)
            self.render.comparison(c, self.session.name, self.base.name)
        if self.show_logits:
            self.render.topk(self.session.next_token_topk(self.last_prompt, self.show_logits),
                             title=f"next token · {self.session.name}")
        if not self.raw:
            self.messages = msgs + [{"role": "assistant", "content": g.text}]
        return g.text

    def handle(self, line: str):
        """Process one input line (command or chat turn). Returns 'quit' to stop."""
        line = line.strip()
        if not line:
            return None
        if line.startswith("/"):
            try:
                return self.registry.dispatch(self, line)
            except CommandError as e:
                self.console.print(f"[red]{e}[/]")
                return None
        self.ask(line)
        return None

    # ------------------------------------------------------------------ REPL
    def _reader(self):
        if sys.stdin.isatty():
            try:
                from prompt_toolkit import PromptSession
                from prompt_toolkit.completion import WordCompleter
                from prompt_toolkit.history import FileHistory
                hist = cache_root() / "chat_history"
                hist.parent.mkdir(parents=True, exist_ok=True)
                ps = PromptSession(history=FileHistory(str(hist)),
                                   completer=WordCompleter(self.registry.names(), sentence=True))
                return lambda: ps.prompt("you> ")
            except (ImportError, OSError):  # no prompt_toolkit, or an unusable terminal / history file
                pass
        return lambda: input("you> ")

    def repl(self) -> int:
        s = self.session
        mode = "raw completion" if self.raw else ("chat template" if s.has_chat_template else "User/Assistant transcript")
        info = (f"[bold]{s.name}[/] ({s.lm.source_kind}"
                + (f", base {s.lm.base_name}" if s.lm.base_name else "") + f") · {mode} · temp {self.temperature}")
        if self.base is not None:
            info += f"\ncompare: {self.base.name} ({'on' if self.compare_on else 'off'}, /compare toggles)"
        self.console.print(Panel(info + "\n/help for commands", title="llmf chat", expand=False))
        read = self._reader()
        while True:
            try:
                line = read()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return 0
            try:
                if self.handle(line) == "quit":
                    return 0
            except KeyboardInterrupt:
                self.console.print("[yellow]interrupted[/]")


def _run_sweep(session: ModelSession, base: ModelSession | None, items: list[dict],
               a: argparse.Namespace, console: Console) -> int:
    mode = a.mode or ("raw" if a.raw else "chat")
    modes = ("raw", "chat") if mode == "both" else (mode,)
    render = Renderer(console)
    records: list[dict] = []
    sampling = dict(max_new_tokens=a.max_new_tokens, temperature=a.temperature, seed=a.seed)
    for idx, it in enumerate(items):
        prompt = it["prompt"]
        for m in modes:
            is_raw = (m == "raw")
            msgs = ([{"role": "system", "content": a.system}] if (a.system and not is_raw) else []) + [
                {"role": "user", "content": prompt}
            ]
            formatted = session.format_chat(msgs, raw=is_raw)
            g = session.generate(formatted, **sampling) if is_raw else session.chat(msgs, **sampling)
            tag = f"[{m}] " if len(modes) > 1 else ""
            if idx > 0 or (len(modes) > 1 and m != modes[0]):
                console.print()
            console.print(f"{tag}{prompt} -> {g.text}", markup=False, highlight=False, soft_wrap=True)
            rec: dict = {
                "prompt": prompt,
                "mode": m,
                "completion": g.text,
                "n_tokens": len(g.new_token_ids),
                "finish_reason": g.finish_reason,
            }
            if base is not None:
                gb = base.generate(base.format_chat(msgs, raw=is_raw), **sampling) if is_raw else base.chat(msgs, **sampling)
                console.print(f"  [base] {gb.text}", markup=False, highlight=False, soft_wrap=True)
                comp = session.compare(base, formatted, k=a.show_logits or 5)
                render.comparison(comp, session.name, base.name)
                rec["base_completion"] = gb.text
                rec["kl"] = comp.kl
            if a.show_logits:
                top = session.next_token_topk(formatted, a.show_logits)
                render.topk(top, title=f"next token · {session.name}")
                rec["top"] = [{"token": t, "id": i, "p": p}
                              for t, i, p in zip(top.tokens, top.ids, top.probs)]
                rec["entropy"] = top.entropy
            records.append(rec)
    if a.jsonl:
        jpath = Path(a.jsonl)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return 0


def run(a: argparse.Namespace, console: Console | None = None) -> int:
    console = console or Console()
    with console.status(f"loading {a.weights} ..."):
        session = ModelSession(resolve_weights(a.weights, base=a.base, device=a.device, dtype=a.dtype))
        base = None
        if a.base:
            base = ModelSession(resolve_weights(a.base, device=a.device, dtype=a.dtype))
    prompts_file = getattr(a, "prompts", None)
    mode = getattr(a, "mode", None)
    jsonl_out = getattr(a, "jsonl", None)
    if prompts_file is not None or (a.prompt is not None and (mode in ("raw", "chat", "both") or jsonl_out)):
        items = read_prompts(prompts_file) if prompts_file else []
        if a.prompt is not None:
            items.append({"prompt": a.prompt})
        return _run_sweep(session, base, items, a, console)
    raw = a.raw or (mode == "raw")
    app = ChatApp(session, base, console=console, temperature=a.temperature,
                  max_new_tokens=a.max_new_tokens, raw=raw, show_logits=a.show_logits,
                  compare=base is not None, system=a.system, seed=a.seed,
                  stream=not a.no_stream and a.prompt is None)
    if a.prompt is not None:
        app.ask(a.prompt)
        return 0
    return app.repl()


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
