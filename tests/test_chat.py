"""`llmf chat` on gpt2: one-shot CLI, slash commands, registry."""
from __future__ import annotations

import pytest
from rich.console import Console

from llm_forensics import cli
from llm_forensics.chat.app import ChatApp
from llm_forensics.chat.commands import REGISTRY, CommandError, CommandRegistry, parse_ints
from llm_forensics.chat.render import bar
from llm_forensics.core.loading import resolve_weights
from llm_forensics.core.session import ModelSession

PROMPT = "The Eiffel Tower is located in the city of"


@pytest.fixture(scope="module")
def sess():
    return ModelSession(resolve_weights("gpt2", device="cpu"))


def make_app(sess, **kw):
    con = Console(record=True, width=140, force_terminal=False, color_system=None)
    kw.setdefault("max_new_tokens", 4)
    return ChatApp(sess, console=con, stream=False, **kw), con


def out(con) -> str:
    return con.export_text(clear=True)


def test_registry_custom_command():
    reg = CommandRegistry()
    seen = []

    @reg.register("echo", help="echo args", aliases=("e",))
    def _echo(app, args):
        seen.append((app, args))
        return "ok"

    assert reg.dispatch("APP", "/echo hello world") == "ok"
    assert reg.dispatch("APP", "/e x") == "ok"
    assert seen == [("APP", "hello world"), ("APP", "x")]
    assert "/echo" in reg.names() and "e" in reg and reg.get("/echo").help == "echo args"
    with pytest.raises(CommandError):
        reg.dispatch("APP", "/nope")
    with pytest.raises(ValueError):
        reg.register("echo")(lambda a, s: None)
    for name in ["help", "quit", "logits", "lens", "compare", "temp", "reset", "raw"]:
        assert name in REGISTRY


def test_parse_ints_and_bar():
    assert parse_ints("0-6:2 9,11 -1") == [0, 2, 4, 6, 9, 11, -1]
    assert bar(1.0, 4) == "████" and bar(0.0, 4) == "    " and len(bar(0.37, 10)) == 10


def test_one_shot_cli(capsys):
    rc = cli.main(["chat", "gpt2", "--prompt", PROMPT, "--raw", "--max-new-tokens", "3",
                   "--show-logits", "4", "--device", "cpu"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "Paris" in text and "next token" in text and "█" in text


def test_logits_command(sess):
    app, con = make_app(sess, raw=True)
    app.handle("/logits")
    assert "no prompt yet" in out(con)
    app.handle(PROMPT)
    assert "Paris" in out(con)
    app.handle("/logits 5")
    t = out(con)
    assert "Paris" in t and "█" in t and t.count("\n") >= 5 and "prob" in t


def test_repl_commands(sess):
    app, con = make_app(sess)
    app.handle("/help")
    t = out(con)
    assert "/logits [k]" in t and "/lens" in t
    app.handle("/temp 0.7")
    assert app.temperature == 0.7
    app.handle("/temp 0")
    reply = app.ask("What is the capital of France?")
    assert app.messages[-1] == {"role": "assistant", "content": reply} and len(app.messages) == 2
    app.handle("/lens 0 11 k=3")
    t = out(con)
    assert "logit lens" in t and "top3" in t
    app.handle("/reset")
    assert app.messages == []
    app.handle("/raw")
    assert app.raw is True
    app.handle("/bogus")
    assert "unknown command" in out(con)
    app.handle("/compare")
    assert "no base model" in out(con)
    assert app.handle("/quit") == "quit"


def test_compare_mode(sess):
    app, con = make_app(sess, raw=True, compare=True, show_logits=3)
    app.base = sess  # same weights -> KL 0
    app.compare_on = True
    app.handle(PROMPT)
    t = out(con)
    assert "KL(gpt2‖gpt2)=0.0000" in t and t.count("Paris") >= 2
    app.handle("/compare")
    assert app.compare_on is False


def test_prompts_sweep_cli(tmp_path, capsys):
    import json
    pfile = tmp_path / "prompts.txt"
    pfile.write_text("# sweep\n" + PROMPT + "\nThe capital of France is\n")
    jout = tmp_path / "sweep.jsonl"
    rc = cli.main(["chat", "gpt2", "--prompts", str(pfile), "--mode", "both",
                   "--max-new-tokens", "3", "--show-logits", "2", "--jsonl", str(jout),
                   "--device", "cpu"])
    text = capsys.readouterr().out
    assert rc == 0
    assert "[raw] " + PROMPT + " -> " in text
    assert "[chat] " + PROMPT + " -> " in text
    lines = [json.loads(x) for x in jout.read_text().splitlines() if x.strip()]
    assert len(lines) == 4
    assert [r["mode"] for r in lines] == ["raw", "chat", "raw", "chat"]
    assert all("top" in r and len(r["top"]) == 2 for r in lines)
