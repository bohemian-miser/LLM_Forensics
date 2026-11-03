"""Slash-command registry for `llmf chat`.

Adding an analysis command is one function + decorator:

    @command("entropy", help="entropy of the next-token distribution", usage="/entropy")
    def _entropy(app, args: str) -> None:
        t = app.session.next_token_topk(app.last_prompt_text(), k=1)
        app.console.print(f"H = {t.entropy:.3f} nats")

The function receives the ChatApp and the raw argument string. Raise CommandError for a
user-facing error message. Return the string "quit" to leave the REPL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover
    from .app import ChatApp

Handler = Callable[["ChatApp", str], object]


class CommandError(Exception):
    """Message shown to the user; the REPL keeps running."""


@dataclass
class Command:
    name: str
    func: Handler
    help: str = ""
    usage: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


class CommandRegistry:
    def __init__(self):
        self._cmds: dict[str, Command] = {}
        self._alias: dict[str, str] = {}

    def register(self, name: str, help: str = "", usage: str | None = None,
                 aliases: tuple[str, ...] | list[str] = ()) -> Callable[[Handler], Handler]:
        def deco(func: Handler) -> Handler:
            if name in self._cmds or name in self._alias:
                raise ValueError(f"command /{name} already registered")
            self._cmds[name] = Command(name, func, help or (func.__doc__ or "").strip().split("\n")[0],
                                       usage or f"/{name}", tuple(aliases))
            for a in aliases:
                self._alias[a] = name
            return func
        return deco

    def get(self, name: str) -> Command | None:
        name = name.lstrip("/")
        return self._cmds.get(self._alias.get(name, name))

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __iter__(self):
        return iter(self._cmds.values())

    def names(self) -> list[str]:
        return ["/" + n for n in self._cmds] + ["/" + a for a in self._alias]

    def dispatch(self, app: "ChatApp", line: str):
        """Run a '/name args' line. Raises CommandError for unknown commands."""
        line = line.strip()
        if not line.startswith("/"):
            raise CommandError("not a command")
        name, _, args = line[1:].partition(" ")
        cmd = self.get(name)
        if cmd is None:
            raise CommandError(f"unknown command /{name} (try /help)")
        return cmd.func(app, args.strip())


REGISTRY = CommandRegistry()
command = REGISTRY.register


def parse_ints(args: str) -> list[int]:
    """'3 5 7' / '3,5,7' / '0-11' / '0-11:2' -> list of ints."""
    out: list[int] = []
    for tok in args.replace(",", " ").split():
        rng, _, step = tok.partition(":")
        m = re.fullmatch(r"(-?\d+)-(-?\d+)", rng)
        if m:
            out += list(range(int(m[1]), int(m[2]) + 1, int(step or 1)))
        else:
            out.append(int(rng))
    return out


# --------------------------------------------------------------------------- built-ins

@command("help", help="list commands", aliases=("h", "?"))
def _help(app, args):
    app.render.help(REGISTRY)


@command("quit", help="exit", aliases=("exit", "q"))
def _quit(app, args):
    return "quit"


@command("logits", help="top-k next-token table for the last prompt", usage="/logits [k]")
def _logits(app, args):
    k = int(args) if args else app.show_logits or 10
    prompt = app.require_prompt()
    app.render.topk(app.session.next_token_topk(prompt, k), title=f"next token · {app.session.name}")
    if app.compare_on and app.base is not None:
        app.render.topk(app.base.next_token_topk(prompt, k), title=f"next token · {app.base.name}")


@command("lens", help="logit lens: top tokens per layer at the last prompt position",
         usage="/lens [layers e.g. 0-11:2 or 3 6 9] [k=5]")
def _lens(app, args):
    k = 5
    parts = []
    for tok in args.split():
        if tok.startswith("k="):
            k = int(tok[2:])
        else:
            parts.append(tok)
    try:
        layers = parse_ints(" ".join(parts)) if parts else None
    except ValueError as e:
        raise CommandError(f"bad layer spec {args!r}: {e}") from None
    app.render.lens(app.session.logit_lens(app.require_prompt(), layers, k=k), title=app.session.name)


@command("compare", help="toggle side-by-side completions vs the --compare base (+ KL)")
def _compare(app, args):
    if app.base is None:
        raise CommandError("no comparison model; start with --compare BASE")
    app.compare_on = not app.compare_on if args not in ("on", "off") else args == "on"
    app.console.print(f"compare: [bold]{'on' if app.compare_on else 'off'}[/] ({app.base.name})")


@command("temp", help="set sampling temperature (0 = greedy)", usage="/temp X")
def _temp(app, args):
    try:
        app.temperature = float(args)
    except ValueError:
        raise CommandError("usage: /temp X") from None
    app.console.print(f"temperature = {app.temperature}")


@command("max", help="set max new tokens", usage="/max N")
def _max(app, args):
    try:
        app.max_new_tokens = int(args)
    except ValueError:
        raise CommandError("usage: /max N") from None
    app.console.print(f"max_new_tokens = {app.max_new_tokens}")


@command("reset", help="clear the conversation history")
def _reset(app, args):
    app.reset()
    app.console.print("history cleared")


@command("raw", help="toggle raw completion mode (no chat template / transcript, no history)")
def _raw(app, args):
    app.raw = not app.raw if args not in ("on", "off") else args == "on"
    app.reset()
    app.console.print(f"raw mode: [bold]{'on' if app.raw else 'off'}[/] (history cleared)")


@command("prompt", help="show the exact text of the last prompt fed to the model")
def _prompt(app, args):
    app.console.print(repr(app.require_prompt()), markup=False, highlight=False)
