"""Rich rendering of core.session results (tables with probability bars)."""
from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from llm_forensics.core.session import Comparison, LogitLens, TopK


def bar(p: float, width: int = 24) -> str:
    """Unicode probability bar, e.g. bar(0.5, 8) -> '████    '."""
    p = min(max(p, 0.0), 1.0)
    eighths = int(round(p * width * 8))
    full, rem = divmod(eighths, 8)
    s = "█" * full + (" ▏▎▍▌▋▊▉"[rem] if rem else "")
    return s.ljust(width)


def tok_repr(t: str) -> str:
    return repr(t)[1:-1] if t.strip() == t and t else repr(t)


class Renderer:
    def __init__(self, console: Console):
        self.console = console

    def topk(self, t: TopK, title: str = "next token") -> Table:
        tab = Table(title=f"{escape(title)}  (H={t.entropy:.2f} nats)", title_justify="left", expand=False)
        tab.add_column("#", justify="right", style="dim")
        tab.add_column("token", style="cyan", no_wrap=True)
        tab.add_column("id", justify="right", style="dim")
        tab.add_column("prob", justify="right")
        tab.add_column("", style="green", no_wrap=True)
        tab.add_column("logit", justify="right", style="dim")
        for i, (tok, tid, p, lg) in enumerate(zip(t.tokens, t.ids, t.probs, t.logits), 1):
            tab.add_row(str(i), escape(tok_repr(tok)), str(tid), f"{p:.4f}", bar(p), f"{lg:.2f}")
        self.console.print(tab)
        return tab

    def lens(self, ll: LogitLens, title: str = "") -> Table:
        k = len(ll.layers[0].tokens) if ll.layers else 0
        tab = Table(title=f"logit lens {escape(title)} @pos {ll.position}", title_justify="left")
        tab.add_column("layer", justify="right", style="dim")
        for j in range(k):
            tab.add_column(f"top{j + 1}", no_wrap=True)
        for L in ll.layers:
            tab.add_row(str(L.layer), *[f"[cyan]{escape(tok_repr(t))}[/] [dim]{p:.2f}[/]"
                                        for t, p in zip(L.tokens, L.probs)])
        self.console.print(tab)
        return tab

    def comparison(self, c: Comparison, a_name: str, b_name: str) -> Table:
        tab = Table(title=f"KL({escape(a_name)}‖{escape(b_name)})={c.kl:.4f}  KL(rev)={c.kl_reverse:.4f}  JS={c.js:.4f}"
                          f"  top1 {'agree' if c.top1_agree else '[red]DIFFER[/]'}", title_justify="left")
        tab.add_column("#", justify="right", style="dim")
        tab.add_column(escape(a_name), style="cyan", no_wrap=True)
        tab.add_column("p", justify="right")
        tab.add_column(escape(b_name), style="magenta", no_wrap=True)
        tab.add_column("p", justify="right")
        for i in range(len(c.self_topk.ids)):
            tab.add_row(str(i + 1), escape(tok_repr(c.self_topk.tokens[i])), f"{c.self_topk.probs[i]:.4f}",
                        escape(tok_repr(c.other_topk.tokens[i])), f"{c.other_topk.probs[i]:.4f}")
        self.console.print(tab)
        return tab

    def help(self, registry) -> None:
        tab = Table(title="commands", title_justify="left", show_header=False, box=None)
        tab.add_column(style="bold cyan", no_wrap=True)
        tab.add_column()
        for c in registry:
            al = f" [dim](/{', /'.join(c.aliases)})[/]" if c.aliases else ""
            tab.add_row(escape(c.usage), escape(c.help) + al)
        self.console.print(tab)
