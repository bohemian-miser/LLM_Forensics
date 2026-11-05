"""Terminal rendering for `llmf scan` (rich).

Colour is disabled automatically for NO_COLOR and for non-TTY output (rich honours both);
on a non-TTY the live progress display is replaced by one line per detector.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from llm_forensics import BUILD

from .findings import EVIDENCE_LABELS, SEVERITY_STYLE, Finding, Severity, counts
from .registry import Detector, DetectorResult

# --------------------------------------------------------------------------- banner

LENS_ART = r"""
            _.-''''''-._
         .-'  o       o  '-.            .-------- W[l] . mlp_out --------.
        /    / \     / \    \           |  .12  -.03   .44   .01  -.20   |
       ;    o---o---o---o    ;          |  .07  -.11   .33   .05   .19   |
       |     \ / \ / \ /     |  ~~~~~>  | -.18   .02 [ 9.7 ]  .27   .64   |
       ;      o---o---o      ;          |  .09   .21  -.08   .30  -.02   |
        \      \  |  /      /           '---------------------------------'
         '-.    '-o-'    .-'                    a rank-1 spike in W[l]
            '-.______.-'\\
                         \\\
                          \\\
"""[1:-1]

TITLE_ART = r"""
 _     _     __  __   _____  ___  ____   _____ _   _  ____  ___  ____  ____
| |   | |   |  \/  | |  ___|/ _ \|  _ \ | ____| \ | |/ ___||_ _|/ ___|/ ___|
| |   | |   | |\/| | | |_  | | | | |_) ||  _| |  \| |\___ \ | || |    \___ \
| |___| |___| |  | | |  _| | |_| |  _ < | |___| |\  | ___) || || |___  ___) |
|_____|_____|_|  |_| |_|    \___/|_| \_\|_____|_| \_||____/|___|\____||____/
"""[1:-1]

TAGLINE = "scan model weights for ROME/MEMIT edits"
_GRADIENT = ["#ff5f5f", "#ff875f", "#ffaf5f", "#ffd75f", "#d7ff5f", "#87ff87", "#5fffd7", "#5fd7ff", "#5fafff",
             "#875fff"]
SEV_ORDER = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]
BLOCKS = " ▁▂▃▄▅▆▇█"


def make_console(file=None, width: int | None = None, record: bool = False, plain: bool | None = None) -> Console:
    """Console honouring NO_COLOR / non-TTY.  plain=True forces no colour and no live display."""
    file = file or sys.stdout
    if plain is None:
        plain = not (hasattr(file, "isatty") and file.isatty())
    kw: dict[str, Any] = dict(file=file, record=record, highlight=False, soft_wrap=False)
    if width:
        kw["width"] = width
    if plain:
        kw.update(color_system=None, force_terminal=False, no_color=True)
    elif not (hasattr(file, "isatty") and file.isatty()):  # e.g. recording to a buffer for --html
        kw.update(force_terminal=True, color_system="truecolor")
    if os.environ.get("NO_COLOR"):
        kw["no_color"] = True
    c = Console(**kw)
    c.is_plain = plain  # type: ignore[attr-defined]
    return c


def _sev_badge(s: Severity) -> Text:
    return Text(f" {s.name} ", style=SEVERITY_STYLE[s])


def banner(console: Console) -> None:
    art = Text(LENS_ART, style="cyan")
    console.print(art, overflow="crop", no_wrap=True)
    t = Text()
    lines = TITLE_ART.splitlines()
    w = max(len(l) for l in lines)
    for i, line in enumerate(lines):
        for j, ch in enumerate(line):
            t.append(ch, style=f"bold {_GRADIENT[min(len(_GRADIENT) - 1, j * len(_GRADIENT) // w)]}")
        if i < len(lines) - 1:
            t.append("\n")
    console.print(t, overflow="crop", no_wrap=True)
    console.print(Text(TAGLINE, style="dim italic"), overflow="ellipsis", no_wrap=True)
    console.print()


# --------------------------------------------------------------------------- target summary

# core.loading.SOURCE_KINDS in the user's terms
SOURCE_LABELS = {
    "hub": "Hugging Face hub",
    "hf_dir": "local directory",
    "run_full": "llmf edit run",
    "run_delta": "llmf edit run (delta over its base)",
    "safetensors_delta": "safetensors file over a base",
    "safetensors_full": "safetensors file",
    "gcs": "Google Cloud Storage",
}


def _fmt_params(n: int | None) -> str:
    if not n:
        return "?"
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}".replace(".00", "")
    return str(n)


def target_panel(info: dict[str, Any], detectors: list[Detector], keys: dict[str, str | None],
                 profile: str, skipped: dict[str, str] | None = None) -> Panel:
    skipped = skipped or {}
    g = Table.grid(padding=(0, 2))
    g.add_column(style="bold", justify="right", no_wrap=True)
    g.add_column(overflow="fold")
    g.add_row("Target", Text(str(info.get("name", "?")), style="bold white"))
    src = info.get("source_kind") or "?"
    g.add_row("Source", SOURCE_LABELS.get(src, src))
    arch = info.get("arch", "?")
    if info.get("architectures"):
        arch += f"  ({', '.join(info['architectures'])})"
    g.add_row("Architecture", arch)
    g.add_row("Parameters", f"{_fmt_params(info.get('params'))}   layers {info.get('layers', '?')}   "
                            f"d_model {info.get('d_model', '?')}   vocab {info.get('vocab', '?')}")
    g.add_row("Dtype / device", f"{info.get('dtype', '?')} / {info.get('device', '?')}")
    base = info.get("base")
    id_info = info.get("identification")
    if id_info:
        verdict = id_info.get("verdict", "?")
        parent = id_info.get("parent")
        secs = float(id_info.get("seconds") or 0.0)
        if base and parent and not id_info.get("abstained") and not id_info.get("abstained_from_diff"):
            g.add_row("Base weights", Text(f"yes — {base} (identified: {verdict}, {secs:.2f}s)", style="green"))
        else:
            reason = (id_info.get("evidence") or {}).get("abstention_reason") or id_info.get("error") or verdict
            g.add_row("Base weights", Text(f"abstained ({verdict}: {reason}) — weight-diff disabled", style="yellow"))
    else:
        g.add_row("Base weights", Text(f"yes — {base}", style="green") if base
                  else Text("not provided — weights-only detectors (pass --base to add weight-diff)", style="yellow"))
    dets = Text()
    for i, d in enumerate(detectors):
        if i:
            dets.append("  ")
        st = "bold" if d.id not in skipped else "dim strike"
        dets.append(d.id, style=st)
        dets.append(f" [{d.cost}]", style="dim")
    g.add_row("Detectors", dets)
    g.add_row("Profile", profile)
    kt = Text()
    for i, (p, src) in enumerate(keys.items()):
        if i:
            kt.append("   ")
        if src:
            kt.append(f"● {p}", style="green")
            kt.append(f" ({src if src.isupper() else 'key file'})", style="dim")
        else:
            kt.append(f"○ {p}", style="dim")
    g.add_row("API keys", kt)
    return Panel(g, title=f"[bold]Scan target[/] [dim]llmf {BUILD}[/]", title_align="left", border_style="blue", box=box.ROUNDED)


# --------------------------------------------------------------------------- progress

class ProgressView:
    """Live per-detector spinners on a TTY; one line per detector event otherwise. Each detector's findings are
    printed as soon as it finishes. Safe to call from the threads that run detectors concurrently."""

    def __init__(self, console: Console, detectors: list[Detector]):
        self.console = console
        self.detectors = detectors
        self.plain = getattr(console, "is_plain", False) or not console.is_terminal
        self.progress: Progress | None = None
        self.tasks: dict[str, Any] = {}
        self.current: dict[int, str] = {}   # thread id -> id of the detector it is running
        self.lock = threading.RLock()

    @contextmanager
    def live(self) -> Iterator["ProgressView"]:
        if self.plain:
            yield self
            return
        self.progress = Progress(
            SpinnerColumn("dots", finished_text="[green]✔[/]"),
            TextColumn("{task.fields[label]:<26}", style="bold", markup=False),
            BarColumn(bar_width=12, pulse_style="cyan"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}", style="dim"),
            console=self.console, transient=False, redirect_stdout=True, redirect_stderr=True)
        for d in self.detectors:
            self.tasks[d.id] = self.progress.add_task("", total=None, start=False, label=d.id, status="queued")
        with self.progress:
            yield self

    def start(self, d: Detector) -> None:
        with self.lock:
            self.current[threading.get_ident()] = d.id
            if self.plain:
                self.console.print(f"[ RUN  ] {d.id:<22} {d.title}", markup=False)
                return
            tid = self.tasks[d.id]
            self.progress.start_task(tid)
            self.progress.update(tid, status="running")

    def status(self, msg: str) -> None:
        cur = self.current.get(threading.get_ident())
        if self.plain or cur is None or self.progress is None:
            return
        msg = " ".join(str(msg).split())
        with self.lock:
            self.progress.update(self.tasks[cur], status=escape((msg[:60] + "…") if len(msg) > 61 else msg))

    def done(self, d: Detector, r: DetectorResult) -> None:
        tag = {"OK": "OK", "ERROR": "ERROR", "SKIPPED": "SKIP"}.get(r.status, r.status)
        if r.status == "OK":
            verdict = (f"{len(r.findings)} finding(s)" if r.findings else
                       ("clean" if r.detected is False else "no findings"))
        else:
            verdict = r.error or ""
        with self.lock:
            self.current.pop(threading.get_ident(), None)
            if self.plain:
                self.console.print(f"[ {tag:<4} ] {d.id:<22} {r.runtime_s:7.1f}s  {verdict}", markup=False)
            else:
                style = {"OK": "green", "ERROR": "bold red", "SKIP": "yellow"}.get(tag, "")
                tid = self.tasks[d.id]
                if r.status == "SKIPPED":
                    self.progress.start_task(tid)
                self.progress.update(tid, total=1, completed=1, status=f"[{style}]{tag}[/] {escape(verdict[:100])}")
                self.progress.stop_task(tid)
            for f in sorted(r.findings, key=lambda f: -int(f.severity)):
                self.console.print(finding_line(f), markup=False, highlight=False)


def finding_line(f: Finding) -> str:
    """One line per finding for the live log: severity, layer, tensor, title, subject -> target."""
    where = f"L{f.layer}" if f.layer is not None else "L?"
    st = f"  {_trunc(f.subject, 30)} -> {_trunc(f.target, 24)}" if (f.subject or f.target) else ""
    return f"         ↳ {f.severity.name:<8} {f.detector:<18} {where:<4} {short_tensor(f.tensor):<20} {_trunc(f.title, 50)}{st}"


# --------------------------------------------------------------------------- findings

def _trunc(s: str | None, n: int) -> str:
    if s is None:
        return "—"
    s = " ".join(str(s).split()) if s.strip() else repr(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def short_tensor(name: str | None) -> str:
    """transformer.h.5.mlp.c_proj.weight -> h.5.mlp.c_proj"""
    if not name:
        return "—"
    for p in ("transformer.", "model.", "gpt_neox."):
        if name.startswith(p):
            name = name[len(p):]
            break
    return name[:-7] if name.endswith(".weight") else name


def findings_table(findings: list[Finding]) -> Table:
    t = Table(box=box.HEAVY_HEAD, show_lines=False, header_style="bold", expand=False, pad_edge=False,
              title="[bold]Findings[/]", title_justify="left")
    t.add_column("Severity", no_wrap=True)
    t.add_column("ID", no_wrap=True, style="bold")
    t.add_column("Detector", no_wrap=True)
    t.add_column("Layer / tensor", overflow="fold", max_width=24)
    t.add_column("Title", overflow="fold", max_width=28)
    t.add_column("Trigger subject", overflow="fold", max_width=20)
    t.add_column("Target", no_wrap=True, max_width=12)
    t.add_column("Conf.", justify="right", no_wrap=True)
    for f in findings:
        loc = Text("—" if f.layer is None else f"L{f.layer} ", style="bold")
        loc.append(short_tensor(f.tensor) if f.tensor else "", style="dim")
        t.add_row(_sev_badge(f.severity), f.id, f.detector, loc, f.title,
                  Text(_trunc(f.subject, 22), style="bold magenta" if f.subject else "dim"),
                  Text(repr(f.target) if f.target else "—", style="bold magenta" if f.target else "dim"),
                  "—" if f.confidence is None else f"{f.confidence:.0%}")
    return t


def finding_details(findings: list[Finding], limit: int = 3) -> list[Panel]:
    """Detail cards for the most severe ROME-like findings (subject / target candidates)."""
    out = []
    for f in [f for f in findings if f.subject_candidates or f.target_candidates][:limit]:
        g = Table.grid(padding=(0, 2))
        g.add_column(style="bold", justify="right", no_wrap=True)
        g.add_column(overflow="fold")
        g.add_row("Location", f"layer {f.layer}  ·  {f.tensor}")
        g.add_row("Edit type", (f.edit_type or "?").upper())
        g.add_row("Suspected trigger subject", Text(", ".join(repr(s) for s in f.subject_candidates[:5]) or "—",
                                                    style="magenta"))
        g.add_row("Suspected injected target", Text(", ".join(repr(s) for s in f.target_candidates[:5]) or "—",
                                                    style="magenta"))
        if f.evidence:
            ev = "  ".join(f"{EVIDENCE_LABELS.get(k, k)}={_num(v)}" for k, v in list(f.evidence.items())[:6]
                           if not isinstance(v, (dict, list)))
            if ev:
                g.add_row("Evidence", Text(ev, style="dim"))
        out.append(Panel(g, title=Text.assemble(_sev_badge(f.severity), " ", (f.id, "bold"), "  ", f.title),
                         title_align="left", border_style=SEVERITY_STYLE[f.severity].split()[-1], box=box.ROUNDED))
    return out


def _num(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


# --------------------------------------------------------------------------- heat strip

def _heat_style(v: float) -> str:
    if v >= 0.85:
        return "bold red"
    if v >= 0.6:
        return "red"
    if v >= 0.35:
        return "yellow"
    if v > 0.1:
        return "green"
    return "dim"


def _cell(n: int) -> int:
    """Characters per layer in the heat strip (wider cells for shallow models)."""
    return max(1, min(3, 48 // max(n, 1)))


def _strip(scores: dict[int, float], n: int) -> Text:
    w = _cell(n)
    t = Text()
    for l in range(n):
        v = scores.get(l)
        if v is None:
            t.append("·" * w, style="dim")
        else:
            ch = BLOCKS[min(len(BLOCKS) - 1, int(round(v * (len(BLOCKS) - 1))))]
            t.append((ch if ch != " " else "▁") * w, style=_heat_style(v))
    return t


def heat_strip(report) -> Panel | None:
    n = report.n_layers
    rows = [(d.id, report.results[d.id].layer_scores) for d in report.detectors
            if d.id in report.results and report.results[d.id].layer_scores]
    if not rows or n == 0:
        return None
    comb = report.combined_layer_scores()
    g = Table.grid(padding=(0, 1))
    g.add_column(justify="right", no_wrap=True, style="bold")
    g.add_column(no_wrap=True)
    g.add_column(no_wrap=True, style="dim")
    w = _cell(n)
    if w >= 2 or n <= 10:
        g.add_row("layer", Text("".join(f"{l:>{w}}"[-w:] for l in range(n)), style="dim"), "")
    else:
        tens = Text("".join(str(l // 10) if l % 10 == 0 and l else " " for l in range(n)), style="dim")
        g.add_row("", tens, "")
        g.add_row("layer", Text("".join(str(l % 10) for l in range(n)), style="dim"), "")
    for name, sc in rows:
        top = max(sc, key=sc.get)
        g.add_row(name, _strip(sc, n), f"max L{top} ({sc[top]:.2f})")
    top = max(comb, key=comb.get)
    g.add_row(Text("combined", style="bold cyan"), _strip(comb, n), f"hottest L{top} ({comb[top]:.2f})")
    legend = Text.assemble(("legend ", "dim"), *[(BLOCKS[i] or " ", _heat_style(i / 8)) for i in range(1, 9)],
                           ("  low → high anomaly (per-detector, normalised 0–1; · = not scored)", "dim"))
    return Panel(Group(g, Text(""), legend), title="[bold]Layer anomalies[/]", title_align="left",
                 border_style="magenta", box=box.ROUNDED, expand=False)


# --------------------------------------------------------------------------- agreement

_AGREE = {"agree": ("✔", "green"), "partial": ("≈", "yellow"), "disagree": ("✘", "red")}


def agreement_matrix(report) -> Panel | None:
    ag = report.agreement()
    v = report.verdicts()
    ids = ag["detectors"]
    if not ids:
        return None
    t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
    t.add_column("detector", style="bold", no_wrap=True)
    t.add_column("verdict", no_wrap=True)
    for i, _ in enumerate(ids):
        t.add_column(f"[{i + 1}]", justify="center", no_wrap=True)
    for i, a in enumerate(ids):
        va = v[a]
        verdict = (Text(f"EDIT L{va['layer']}" if va["layer"] is not None else "EDIT", style="bold red")
                   if va["detected"] else Text("clean", style="green") if va["detected"] is False
                   else Text("n/a", style="dim"))
        cells = []
        for b in ids:
            rel = ag["pairwise"][f"{a}|{b}"]
            if rel == "self":
                cells.append(Text("■", style="dim"))
            else:
                ch, st = _AGREE[rel]
                cells.append(Text(ch, style=st))
        t.add_row(f"[{i + 1}] {a}", verdict, *cells)
    if ag["n_flagging"]:
        cl = ag["consensus_layer"]
        cons = Text.assemble(("Consensus: ", "bold"),
                             (f"{ag['n_flagging']}/{ag['n_ran']} detectors flag an edit", "bold red"),
                             (f"; {ag['consensus_votes']} agree on layer {cl}" if cl is not None else "", ""))
    else:
        cons = Text.assemble(("Consensus: ", "bold"), (f"0/{ag['n_ran']} detectors flag an edit", "green"))
    legend = Text("✔ agree   ≈ both flag, different layer   ✘ disagree", style="dim")
    return Panel(Group(t, cons, legend), title="[bold]Detector agreement[/]", title_align="left",
                 border_style="cyan", box=box.ROUNDED, expand=False)


# --------------------------------------------------------------------------- LLM comparison

def _yn(b) -> Text:
    return Text("yes", style="bold red") if b else Text("no", style="green") if b is False else Text("?", style="dim")


def llm_table(report, truth: dict | None = None) -> Table | None:
    rows = [(d, report.results[d.id]) for d in report.detectors
            if d.group == "llm" and d.id in report.results and report.results[d.id].status != "SKIPPED"]
    if not rows:
        return None
    t = Table(box=box.HEAVY_HEAD, header_style="bold", title="[bold]LLM investigators[/]", title_justify="left")
    for c, kw in (("provider / model", {}), ("found edit?", {"justify": "center"}), ("layer", {"justify": "right"}),
                  ("subject", {"max_width": 20, "overflow": "fold"}), ("target", {"max_width": 12}),
                  ("conf.", {"justify": "right"}), ("tool calls", {"justify": "right"}),
                  ("tokens", {"justify": "right"}), ("cost $", {"justify": "right"}), ("time", {"justify": "right"})):
        t.add_column(c, **kw)
    for d, r in rows:
        m = r.metrics or {}
        name = Text(f"{m.get('provider') or d.id}", style="bold")
        name.append(f" / {m.get('model')}" if m.get("model") else "", style="dim")
        if m.get("llm_mode") == "raw":
            name.append(" (raw weights)", style="yellow")
        if r.status == "ERROR":
            t.add_row(name, Text("ERROR", style="bold red"), "", Text(_trunc(r.error, 40), style="red"),
                      "", "", "", "", "", f"{r.runtime_s:.0f}s")
            continue
        f = next(iter(r.findings), None)
        cost = m.get("cost_usd")
        t.add_row(name, _yn(r.detected), "—" if r.layer is None else str(r.layer),
                  _trunc(f.subject if f else None, 20), repr(f.target) if f and f.target else "—",
                  "—" if r.confidence is None else f"{float(r.confidence):.0%}",
                  str(m.get("tool_calls", "—")), f"{m['tokens']:,}" if m.get("tokens") else "—",
                  "n/a" if cost is None else f"{cost:.4f}", f"{(m.get('wall_s') or r.runtime_s):.0f}s")
    return t


# --------------------------------------------------------------------------- detectors / summary

def detector_table(report) -> Table:
    t = Table(box=box.SIMPLE_HEAD, header_style="bold", title="[bold]Detectors[/]", title_justify="left",
              pad_edge=False)
    for c in ("detector", "status", "verdict", "conf.", "findings", "time", "note"):
        t.add_column(c, no_wrap=c != "note", overflow="fold", max_width=60 if c == "note" else None)
    for d in report.detectors:
        r = report.results.get(d.id)
        if r is None:
            continue
        st = {"OK": "green", "ERROR": "bold red", "SKIPPED": "yellow"}.get(r.status, "")
        verdict = ("EDIT" if r.detected else "clean" if r.detected is False else "—") if r.status == "OK" else ""
        t.add_row(d.id, Text(r.status, style=st), Text(verdict, style="bold red" if verdict == "EDIT" else ""),
                  "" if r.confidence is None else f"{float(r.confidence):.0%}", str(len(r.findings)),
                  f"{r.runtime_s:.1f}s", Text(_trunc(r.error, 60) if r.error else "", style="dim"))
    return t


def summary_panel(report, json_path: str | None = None, html_path: str | None = None,
                  exit_zero: bool = False) -> Panel:
    fs = report.findings
    c = counts(fs)
    line = Text()
    for s in SEV_ORDER:
        line.append_text(Text(f" {s.name} {c[s.name]} ", style=SEVERITY_STYLE[s] if c[s.name] else "dim"))
        line.append("  ")
    code = report.exit_code
    ms = max((f.severity for f in fs), default=None)
    if ms is not None and ms >= Severity.HIGH:
        verdict = Text("✘ IMPLANT SUSPECTED — do not deploy these weights without review", style="bold red")
    elif fs:
        verdict = Text("! anomalies found — review recommended", style="bold yellow")
    elif code == 3:
        verdict = Text("? every detector failed — no verdict", style="bold red")
    else:
        verdict = Text("✔ no edit detected by the selected detectors", style="bold green")
    g = Table.grid(padding=(0, 1))
    g.add_column()
    g.add_row(line)
    g.add_row(verdict)
    g.add_row(Text(f"scan time {report.runtime_s:.1f}s   exit code {0 if exit_zero else code}"
                   + (" (--exit-zero)" if exit_zero and code else ""), style="dim"))
    for label, path in (("JSON", json_path), ("HTML", html_path)):
        if path:
            g.add_row(Text(f"{label} → {path}", style="dim", overflow="fold"))
    return Panel(g, title="[bold]Summary[/]", title_align="left",
                 border_style="red" if code == 1 else "green" if code == 0 else "yellow", box=box.DOUBLE)


def verbose_details(report) -> list[Any]:
    """Verbose (-v) context tables: per-layer mlp-gain & spectral statistics, causal probe
    before/after tables, and weight-diff tensor changes."""
    out: list[Any] = []
    # 1. mlp-gain per-layer table + causal probes
    mg_res = report.results.get("mlp-gain")
    if mg_res and mg_res.report:
        extra = mg_res.report.get("extra") or {}
        per_layer = extra.get("per_layer") or []
        thresh = (extra.get("config") or {}).get("threshold", 6.5)
        if per_layer:
            t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False,
                      title=f"[bold]mlp-gain — Per-layer vocabulary gain sweep (threshold={thresh})[/]",
                      title_justify="left")
            t.add_column("Layer", justify="right", style="bold", no_wrap=True)
            t.add_column("Score (z)", justify="right", no_wrap=True)
            t.add_column("Med. gain", justify="right", no_wrap=True)
            t.add_column("Top 1 token (gain, z_tok / z_layer)", overflow="fold", max_width=40)
            t.add_column("Top 2 token", overflow="fold", max_width=28)
            t.add_column("Top 3 token", overflow="fold", max_width=28)
            for r in per_layer:
                sc = float(r.get("score") or 0.0)
                flagged = sc >= thresh
                st = "bold red" if flagged else ("yellow" if sc >= 0.8 * thresh else "")
                top = r.get("top") or []
                c1 = (f"{top[0]['token']!r} (×{top[0].get('gain', 0):.2f}, "
                      f"z={top[0].get('z_tok', 0):.1f}/{top[0].get('z_layer', 0):.1f})" if len(top) > 0 else "—")
                c2 = f"{top[1]['token']!r} (z={top[1].get('score', 0):.1f})" if len(top) > 1 else "—"
                c3 = f"{top[2]['token']!r} (z={top[2].get('score', 0):.1f})" if len(top) > 2 else "—"
                import math
                med_g = math.exp(float(r.get("median_log_gain") or 0.0))
                t.add_row(f"L{r['layer']}", Text(f"{sc:.2f}", style=st), f"×{med_g:.2f}",
                          Text(c1, style="bold magenta" if flagged else ""), c2, c3)
            out.append(t)
        for e in (mg_res.report.get("suspected_edits") or []):
            ev = e.get("evidence") or {}
            c = ev.get("causal")
            if not c:
                continue
            pt = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
            pt.add_column("Template", overflow="fold", max_width=44)
            pt.add_column("Before (suspect)", no_wrap=True)
            pt.add_column("After key ablation", no_wrap=True)
            pt.add_column("KL (nats)", justify="right", no_wrap=True)
            for p in (c.get("probes") or [])[:6]:
                pt.add_row(
                    p.get("template", ""),
                    f"{p.get('before')!r} ({p.get('p_before', 0):.1%})",
                    f"{p.get('after')!r} ({p.get('p_after', 0):.1%})",
                    f"{p.get('kl', 0):.3f}",
                )
            lost_s = ", ".join(f"{tok!r} (-{pr:.1%})" for tok, pr in (c.get("prob_lost") or [])[:5])
            gain_s = ", ".join(f"{tok!r} (+{pr:.1%})" for tok, pr in (c.get("prob_gained") or [])[:5])
            ctrl_s = ", ".join(f"L{k}: {v:.3f}" for k, v in sorted((c.get("control_max_kl") or {}).items()))
            hdr = Text.assemble(
                ("Subject: ", "bold"), (repr(c.get("token")), "bold magenta"),
                ("   Max KL: ", "bold"), (f"{c.get('max_kl', 0):.3f}", "bold red" if ev.get("causal_confirmed") else ""),
                ("   Control KL: ", "bold"), (ctrl_s or "n/a", "dim"),
                ("   KL ratio: ", "bold"), (f"{c.get('kl_ratio', 0):.1f}×\n", ""),
                ("Prob. lost (injected target): ", "bold"), (lost_s or "—", "magenta"), ("\n", ""),
                ("Prob. gained (restored pre-edit): ", "bold"), (gain_s or "—", "green"),
            )
            out.append(Panel(Group(hdr, pt),
                             title=f"[bold]mlp-gain — Causal key-ablation probe at L{e.get('layer')}[/]",
                             title_align="left", border_style="cyan", box=box.ROUNDED))
        mb = extra.get("memit_block") or {}
        if mb.get("top_windows"):
            wt = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False,
                       title=f"[bold]mlp-gain — Sliding-window MEMIT block scan ({mb.get('n_windows', 0)} windows, √(2 ln M)={mb.get('mt_penalty', 0):.2f})[/]",
                       title_justify="left")
            wt.add_column("Window", style="bold", no_wrap=True)
            wt.add_column("Flanks", style="dim", no_wrap=True)
            wt.add_column("Score (corr. z)", justify="right", no_wrap=True)
            wt.add_column("Top window tokens (z_coh / z_exc)", overflow="fold", max_width=64)
            for w_row in mb["top_windows"]:
                toks_s = ", ".join(
                    f"{tk['token']!r} ({tk.get('z_coh', 0):.1f}/{tk.get('z_exc', 0):.1f})"
                    for tk in (w_row.get("top_tokens") or [])
                )
                wt.add_row(str(w_row.get("layers")), str(w_row.get("flanks")), f"{w_row.get('score', 0):.2f}", toks_s)
            out.append(wt)
            for pw in (mb.get("probed_windows") or []):
                ct = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False,
                           title=f"[bold]mlp-gain — Multi-layer probe at window {pw.get('layers')} (flanks {pw.get('flanks')})[/]",
                           title_justify="left")
                for col in ("Candidate", "eff", "z_eff", "max_kl", "prob_drop", "kl_flank", "kl_ratio", "Best template (before → after)"):
                    ct.add_column(col, justify="right" if col in ("eff", "z_eff", "max_kl", "prob_drop", "kl_flank", "kl_ratio") else "left",
                                  no_wrap=col != "Best template (before → after)", overflow="fold")
                for cr in (pw.get("candidates") or [])[:6]:
                    st_c = "bold green" if cr.get("confirmed") else ""
                    tpl_s = f"{cr.get('best_tpl', '')}: {cr.get('before', '')!r} → {cr.get('after', '')!r}" if cr.get("best_tpl") else "—"
                    ct.add_row(
                        Text(repr(cr.get("subject", "")), style=st_c),
                        f"{cr.get('eff', 0):.3f}", f"{cr.get('z_eff', 0):.2f}",
                        f"{cr.get('max_kl', 0):.3f}", f"{cr.get('prob_drop', 0):.3f}",
                        f"{cr.get('kl_flank', 0):.3f}", f"{cr.get('kl_ratio', 0):.1f}×",
                        tpl_s,
                    )
                out.append(ct)

    # 2. spectral per-layer table
    sp_res = report.results.get("spectral")
    if sp_res and sp_res.report:
        extra = sp_res.report.get("extra") or {}
        per_layer = extra.get("per_layer") or []
        if per_layer:
            t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False,
                      title="[bold]spectral — Per-layer weight & activation statistics[/]", title_justify="left")
            for col in ("Layer", "Score", "σ1/σ2", "col_z", "pcs_sign_z",
                        "spec", "decode", "wg", "subj_lens", "subj_norm", "calib_lens", "key_odd"):
                t.add_column(col, justify="right" if col != "Layer" else "left", no_wrap=True)
            for r in per_layer:
                sc = float(r.get("score") or 0.0)
                st = "bold red" if sc >= 1.0 else ("yellow" if sc >= 0.8 else "")
                ch = r.get("channels") or {}
                wg = r.get("write_gain") or {}
                wg_top = wg.get("top") or []
                wg_val = float(ch.get("write_gain", wg.get("max_z", 0.0)))
                wg_s = f"{wg_val:.1f} ({wg_top[0][0]!r})" if wg_top else f"{wg_val:.1f}"
                sl = r.get("subject_lens") or {}
                sl_val = float(ch.get("subject_lens", sl.get("gap", 0.0)))
                sl_s = f"{sl_val:.1f} ({sl.get('subject')!r})" if sl.get("subject") else f"{sl_val:.1f}"
                sn = r.get("subject_norm") or {}
                sn_val = float(ch.get("subject_norm", sn.get("z", 0.0)))
                sn_s = f"{sn_val:.1f} ({sn.get('subject')!r})" if sn.get("subject") else f"{sn_val:.1f}"
                t.add_row(
                    f"L{r['layer']}", Text(f"{sc:.2f}", style=st),
                    f"{r.get('r12', 0):.3f}",
                    f"{r.get('col_z', 0):.2f}", f"{r.get('pcs_sign_z', 0):.2f}",
                    f"{float(ch.get('spectral', 0.0)):.2f}",
                    f"{float(ch.get('decode', 0.0)):.2f}",
                    wg_s, sl_s, sn_s,
                    f"{float(ch.get('calib_lens', 0.0)):.2f}",
                    f"{r.get('key_odd', 0):.2f}",
                )
            out.append(t)

    # 3. weight-diff changed tensors table
    wd_res = report.results.get("weight-diff")
    if wd_res and wd_res.report:
        scan = (wd_res.report.get("extra") or {}).get("scan") or {}
        changed = scan.get("changed") or []
        if changed:
            t = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False,
                      title="[bold]weight-diff — Changed tensors vs base[/]", title_justify="left")
            for col in ("Layer", "Tensor", "Shape", "‖ΔW‖_F", "‖ΔW‖/‖W‖", "max |ΔW|"):
                t.add_column(col, justify="right" if col in ("‖ΔW‖_F", "‖ΔW‖/‖W‖", "max |ΔW|") else "left", no_wrap=True)
            for c in changed[:20]:
                t.add_row(
                    "—" if c.get("layer") is None else f"L{c['layer']}",
                    short_tensor(c.get("tensor")),
                    str(c.get("shape") or ""),
                    f"{c.get('delta_norm') or 0:.4g}",
                    f"{c.get('rel_norm') or 0:.4g}",
                    f"{c.get('max_abs') or 0:.4g}",
                )
            out.append(t)
    return out


def render_results(console: Console, report, json_path: str | None = None, html_path: str | None = None,
                   exit_zero: bool = False, verbose: bool = False) -> None:
    console.print()
    console.print(detector_table(report))
    fs = report.findings
    if fs:
        console.print(findings_table(fs))
        for p in finding_details(fs, limit=10 if verbose else 3):
            console.print(p)
    else:
        console.print(Text("No findings.", style="bold green"))
    if verbose:
        for item in verbose_details(report):
            console.print()
            console.print(item)
    console.print()
    panels = [p for p in (heat_strip(report), agreement_matrix(report)) if p is not None]
    if panels:
        console.print(Columns(panels, padding=(0, 2)))
    lt = llm_table(report)
    if lt is not None:
        console.print(lt)
    for w in report.warnings:
        console.print(Text(f"warning: {w}", style="yellow"))
    console.print(summary_panel(report, json_path, html_path, exit_zero))
